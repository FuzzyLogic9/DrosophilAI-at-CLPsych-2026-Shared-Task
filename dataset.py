import os
import json
import pandas as pd
from glob import glob
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PresenceRatingDataset(Dataset):
    """Dataset for presence rating with Mental-RoBERTa"""
    
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract presence scores from evidence
        self.df['adaptive_score'] = self.df['evidence'].apply(
            lambda x: x.get('adaptive-state', {}).get('Presence', 1)
        )
        self.df['maladaptive_score'] = self.df['evidence'].apply(
            lambda x: x.get('maladaptive-state', {}).get('Presence', 1)
        )
        
        print(f"\n=== Presence Score Distribution ===")
        print(f"Adaptive scores:")
        print(self.df['adaptive_score'].value_counts().sort_index())
        print(f"\nMaladaptive scores:")
        print(self.df['maladaptive_score'].value_counts().sort_index())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            row['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'adaptive_score': torch.tensor(row['adaptive_score'], dtype=torch.float),
            'maladaptive_score': torch.tensor(row['maladaptive_score'], dtype=torch.float),
            'post_id': row['post_id'],
            'timeline_id': row['timeline_id']
        }

class CLPsychDataLoader:
    """Load CLPsych data with proper ordering"""
    
    def __init__(self, input_dir, split='train'):

        self.split = split
        if split == 'train':
            self.input_dir = os.path.join(input_dir, 'train')
        elif split == 'val':
            self.input_dir = os.path.join(input_dir, 'valid')
        elif split == 'test':
            self.input_dir = os.path.join(input_dir, 'test')
        else:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")
        self.df = None

    def load_clpsych_data(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            id, post = data['timeline_id'], data['posts']
        return id, post

    def load(self):
        """Load and parse JSON data into sorted DataFrame"""

        train_posts = []
        for file in glob(os.path.join(self.input_dir, '*.json')):
            # print(f"Loading {file}...")
            id, posts = self.load_clpsych_data(file)
            print(f"Loaded {id} with {len(posts)} posts.")
            for post in posts:
                # print(post)
                try:
                    assert 'post_id' in post
                    assert 'post' in post
                    assert 'evidence' in post
                except AssertionError:
                    print(f"Timeline {id}, Post {post['post_id']} is missing required fields.")
                    continue
                train_posts.append({
                    'timeline_id': id,
                    'post_id': post['post_id'],
                    'post_index': post['post_index'],
                    'text': post['post'],
                    'well_being': post.get('Well-being', 0),
                    'is_switch': post.get('Switch', 0),
                    'is_escalation': post.get('Escalation', 0),
                    'evidence': post['evidence']
                })
        
        # Create DataFrame and sort by timeline_id and post_index
        self.df = pd.DataFrame(train_posts)
        self.df = self.df.sort_values(['timeline_id', 'post_index'])
        
        print(f"Loaded {len(self.df)} posts from {self.df['timeline_id'].nunique()} timelines")
        
        return self.df
    
    def verify_order(self):
        """Verify posts are in correct order within each timeline"""
        print("\n=== Verifying Post Order ===")
        issues = []
        
        for timeline_id in self.df['timeline_id'].unique():
            timeline_posts = self.df[self.df['timeline_id'] == timeline_id]
            indices = timeline_posts['post_index'].tolist()
            
            # Check if indices are in ascending order
            if indices != sorted(indices):
                issues.append(f"Timeline {timeline_id}: {indices}")
        
        if issues:
            print("❌ Order issues found:")
            for issue in issues:
                print(f"  {issue}")
            return False
        else:
            print("✅ All posts are in correct order")
            return True
    
    def get_stats(self):
        """Print dataset statistics"""
        print("\n=== Dataset Statistics ===")
        print(f"Total timelines: {self.df['timeline_id'].nunique()}")
        print(f"Total posts: {len(self.df)}")
        print(f"Avg posts per timeline: {len(self.df) / self.df['timeline_id'].nunique():.2f}")
        
        # print(f"\nSwitch events: {self.df['is_switch'].sum()} ({self.df['is_switch'].mean()*100:.1f}%)")
        # print(f"Escalation events: {self.df['is_escalation'].sum()} ({self.df['is_escalation'].mean()*100:.1f}%)")
        
        # ABCD presence
        print("\n=== ABCD Element Presence ===")
        adaptive_counts = defaultdict(int)
        maladaptive_counts = defaultdict(int)
        
        for _, row in self.df.iterrows():
            evidence = row['evidence']
            
            # Adaptive
            if 'adaptive-state' in evidence:
                for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']:
                    if dim in evidence['adaptive-state'] and evidence['adaptive-state'][dim].get('Category'):
                        adaptive_counts[dim] += 1
            
            # Maladaptive
            if 'maladaptive-state' in evidence:
                for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']:
                    if dim in evidence['maladaptive-state'] and evidence['maladaptive-state'][dim].get('Category'):
                        maladaptive_counts[dim] += 1
        
        print("\nAdaptive:")
        for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']:
            print(f"  {dim}: {adaptive_counts[dim]}")
        
        print("\nMaladaptive:")
        for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']:
            print(f"  {dim}: {maladaptive_counts[dim]}")


if __name__ == "__main__":
    # Load data
    train_loader = CLPsychDataLoader('tasks12/', split='train')
    val_loader = CLPsychDataLoader('tasks12/', split='val')
    # test_loader = CLPsychDataLoader('tasks12/', split='test')
    val_df = val_loader.load()
    val_loader.verify_order()
    val_loader.get_stats()


# for i in range(min(3, len(val_df))):
#     print(f"\n--- Example {i+1} ---")
#     evidence = val_df.iloc[i]['evidence']
#     print(f"Post: {val_df.iloc[i]['text'][:80]}...")
#     print(f"\nEvidence keys: {evidence.keys()}")
    
#     # Check adaptive-state
#     if 'adaptive-state' in evidence:
#         print(f"\nAdaptive-state keys: {evidence['adaptive-state'].keys()}")
#         print(f"Adaptive-state content:")
#         for key, value in evidence['adaptive-state'].items():
#             print(f"  {key}: {value}")
    
#     # Check maladaptive-state
#     if 'maladaptive-state' in evidence:
#         print(f"\nMaladaptive-state keys: {evidence['maladaptive-state'].keys()}")
#         print(f"Maladaptive-state content:")
#         for key, value in evidence['maladaptive-state'].items():
#             print(f"  {key}: {value}")
    
#     print("\n" + "="*60)


"""
Prepare instruction tuning data for Task 1.2 (Presence Rating)
"""


def create_presence_instruction(post_text, evidence):
    """
    Create instruction-tuning example for presence rating
    
    Args:
        post_text: The social media post
        evidence: ABCD evidence with presence scores
    
    Returns:
        Dict with instruction, input, output
    """
    
    instruction = """You are an expert in mental health text analysis using the MIND framework. Analyze the social media post and rate the presence of adaptive and maladaptive self-states on a scale of 1-5.

    Rating Scale:
    1 - Not present: The self-state is not expressed in the post.
    2 - Somewhat present: The self-state is expressed, but plays a subtle, limited role.
    3 - Moderately present: The self-state is clearly expressed and moderately contributes.
    4 - Much present: The self-state strongly influences and shapes the experience.
    5 - Highly present: The self-state strongly shapes and clearly defines the overall experience.

    Analyze the post and provide:
    1. Brief analysis of adaptive elements
    2. Adaptive presence score (1-5)
    3. Brief analysis of maladaptive elements
    4. Maladaptive presence score (1-5)
    5. Overall reasoning

    Output in JSON format."""

    # Extract presence scores
    adaptive_score = evidence.get('adaptive-state', {}).get('Presence', 1)
    maladaptive_score = evidence.get('maladaptive-state', {}).get('Presence', 1)
    
    # Generate analysis text from ABCD elements
    adaptive_analysis = generate_adaptive_analysis(evidence, adaptive_score)
    maladaptive_analysis = generate_maladaptive_analysis(evidence, maladaptive_score)
    reasoning = generate_reasoning(evidence, adaptive_score, maladaptive_score)
    
    # Output JSON
    output = {
        "adaptive_analysis": adaptive_analysis,
        "adaptive_score": adaptive_score,
        "maladaptive_analysis": maladaptive_analysis,
        "maladaptive_score": maladaptive_score,
        "reasoning": reasoning
    }
    
    return {
        "instruction": instruction,
        "input": f"Post: {post_text}",
        "output": json.dumps(output, indent=2)
    }


def generate_adaptive_analysis(evidence, score):
    """Generate adaptive analysis text from ABCD elements"""
    
    adaptive_state = evidence.get('adaptive-state', {})
    elements = []
    
    # Extract adaptive ABCD elements
    for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']:
        if dim in adaptive_state and adaptive_state[dim].get('Category'):
            category = adaptive_state[dim]['Category']
            evidence_text = adaptive_state[dim].get('highlighted_evidence', '')
            if evidence_text:
                elements.append(f"{dim} ({category}): '{evidence_text}'")
            else:
                elements.append(f"{dim}: {category}")
    
    # Generate text based on score
    if score == 5:
        if elements:
            return f"Highly present. Adaptive self-states clearly define the experience: {'; '.join(elements)}. These elements dominate the post."
        else:
            return "Highly present with strong adaptive orientation throughout."
    elif score == 4:
        if elements:
            return f"Much present. Adaptive self-states strongly shape the post: {'; '.join(elements)}. These elements significantly influence the experience."
        else:
            return "Much present with clear adaptive elements."
    elif score == 3:
        if elements:
            return f"Moderately present. Adaptive elements include: {'; '.join(elements)}. These contribute to the overall experience."
        else:
            return "Moderately present with some adaptive elements."
    elif score == 2:
        if elements:
            return f"Somewhat present. Subtle adaptive elements: {'; '.join(elements)}. These play a limited role."
        else:
            return "Somewhat present but subtle."
    else:  # score == 1
        return "Not present. No clear adaptive self-states expressed."


def generate_maladaptive_analysis(evidence, score):
    """Generate maladaptive analysis text from ABCD elements"""
    
    maladaptive_state = evidence.get('maladaptive-state', {})
    elements = []
    
    # Extract maladaptive ABCD elements
    for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']:
        if dim in maladaptive_state and maladaptive_state[dim].get('Category'):
            category = maladaptive_state[dim]['Category']
            evidence_text = maladaptive_state[dim].get('highlighted_evidence', '')
            if evidence_text:
                elements.append(f"{dim} ({category}): '{evidence_text}'")
            else:
                elements.append(f"{dim}: {category}")
    
    # Generate text based on score
    if score == 5:
        if elements:
            return f"Highly present. Maladaptive self-states clearly define the experience: {'; '.join(elements)}. These patterns dominate."
        else:
            return "Highly present with strong maladaptive patterns."
    elif score == 4:
        if elements:
            return f"Much present. Maladaptive self-states strongly shape the post: {'; '.join(elements)}. These significantly influence the experience."
        else:
            return "Much present with clear maladaptive elements."
    elif score == 3:
        if elements:
            return f"Moderately present. Maladaptive elements include: {'; '.join(elements)}. These contribute to the experience."
        else:
            return "Moderately present with some maladaptive elements."
    elif score == 2:
        if elements:
            return f"Somewhat present. Subtle maladaptive elements: {'; '.join(elements)}. These play a limited role."
        else:
            return "Somewhat present but subtle."
    else:  # score == 1
        return "Not present. No clear maladaptive self-states expressed."


def generate_reasoning(evidence, adaptive_score, maladaptive_score):
    """Generate overall reasoning"""
    
    adaptive_state = evidence.get('adaptive-state', {})
    maladaptive_state = evidence.get('maladaptive-state', {})
    
    adaptive_count = sum(1 for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D'] 
                         if dim in adaptive_state and adaptive_state[dim].get('Category'))
    maladaptive_count = sum(1 for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D'] 
                            if dim in maladaptive_state and maladaptive_state[dim].get('Category'))
    
    score_map = {1: "not present", 2: "subtly present", 3: "moderately present", 
                 4: "strongly present", 5: "dominantly present"}
    
    reasoning = f"The post shows {score_map[adaptive_score]} adaptive self-states ({adaptive_count} elements detected) and {score_map[maladaptive_score]} maladaptive self-states ({maladaptive_count} elements detected)."
    
    # Add context about dominance
    if adaptive_score > maladaptive_score + 1:
        reasoning += " The adaptive orientation clearly dominates the experience."
    elif maladaptive_score > adaptive_score + 1:
        reasoning += " The maladaptive patterns clearly dominate the experience."
    elif adaptive_score == maladaptive_score:
        reasoning += " Both adaptive and maladaptive states are equally present."
    else:
        reasoning += " The post shows a mix of both states."
    
    return reasoning


def prepare_instruction_dataset(data_path, split='train'):
    """Prepare complete instruction dataset"""
    
    print(f"\n{'='*60}")
    print(f"Preparing Instruction Dataset for {split}")
    print('='*60)
    
    # Load data
    loader = CLPsychDataLoader(data_path, split=split)
    df = loader.load()
    
    print(f"\nLoaded {len(df)} posts")
    
    # Create instruction examples
    instruction_data = []
    
    for idx, row in df.iterrows():
        example = create_presence_instruction(row['text'], row['evidence'])
        
        # Add metadata
        example['timeline_id'] = row['timeline_id']
        example['post_id'] = row['post_id']
        example['post_index'] = row['post_index']
        
        instruction_data.append(example)
    
    print(f"Created {len(instruction_data)} instruction examples")
    
    # Save
    output_file = f'{split}_presence_instructions.json'
    with open(output_file, 'w') as f:
        json.dump(instruction_data, f, indent=2)
    
    print(f"✅ Saved to {output_file}")
    
    return instruction_data


