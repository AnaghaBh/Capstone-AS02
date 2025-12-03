# Psychological Frameworks for Misinformation Analysis

This document describes the psychological frameworks used in the misinformation classifier and how they map to the model's output labels.

## Framework 1: Elaboration Likelihood Model (ELM)

The Elaboration Likelihood Model, developed by Petty and Cacioppo (1986), describes two routes to persuasion:

### Central Route Processing (`central_route_present`)

**Definition:** Systematic, effortful processing where individuals carefully evaluate the quality of arguments and evidence.

**Characteristics:**
- Focus on argument strength and logical reasoning
- Detailed analysis of evidence and facts
- Critical evaluation of source credibility
- Consideration of counterarguments

**Misinformation Cues:**
- Appeals to scientific studies or research
- Detailed explanations or technical language
- Citations of experts or authorities
- Logical argument structure
- Statistical claims or data presentation

**Example Headlines:**
- "New peer-reviewed study shows 95% effectiveness rate"
- "Leading researchers at Harvard confirm breakthrough discovery"
- "Clinical trial data reveals significant improvement in patients"

### Peripheral Route Processing (`peripheral_route_present`)

**Definition:** Heuristic processing where individuals rely on superficial cues rather than argument quality.

**Characteristics:**
- Quick, automatic judgments
- Reliance on simple decision rules
- Emotional responses over rational analysis
- Attention to surface features

**Misinformation Cues:**
- Emotional language (fear, excitement, urgency)
- Authority claims without evidence ("doctors hate this")
- Social proof ("millions of people can't be wrong")
- Attractive presentation or celebrity endorsement
- Simple, catchy phrases or slogans

**Example Headlines:**
- "Doctors hate this one weird trick!"
- "Celebrities swear by this miracle cure"
- "You won't believe what happens next"
- "This will shock you!"

## Framework 2: Cognitive Biases

### Naturalness Bias (`naturalness_bias`)

**Definition:** The tendency to prefer "natural" products, treatments, or solutions over artificial or synthetic alternatives, often without scientific justification.

**Characteristics:**
- Assumption that natural = safe/effective
- Distrust of synthetic or manufactured products
- Appeal to nature fallacy
- Romanticization of traditional methods

**Misinformation Cues:**
- Emphasis on "natural," "organic," "herbal"
- Contrast with "artificial," "chemical," "synthetic"
- References to traditional or ancient wisdom
- Claims about purity or being "chemical-free"

**Example Headlines:**
- "Ancient herbal remedy outperforms modern medicine"
- "Natural cure doctors don't want you to know"
- "Chemical-free solution to all your problems"
- "Pure, organic treatment with no side effects"

### Availability Bias (`availability_bias`)

**Definition:** The tendency to overestimate the likelihood or importance of events based on how easily examples come to mind.

**Characteristics:**
- Overemphasis on recent or memorable events
- Vivid, dramatic examples over statistical reality
- Personal anecdotes treated as universal truth
- Media coverage influencing perceived frequency

**Misinformation Cues:**
- Dramatic personal stories or testimonials
- References to recent, highly publicized events
- Vivid, memorable imagery or scenarios
- "Everyone knows" or "it's happening everywhere"
- Anecdotal evidence presented as proof

**Example Headlines:**
- "Local mom discovers cure after tragic diagnosis"
- "This happened to my neighbor - it could happen to you"
- "Shocking incident reveals hidden danger in your home"
- "You've probably experienced this without knowing"

### Illusory Correlation (`illusory_correlation`)

**Definition:** The tendency to perceive relationships between variables when no such relationship exists, or to overestimate the strength of weak relationships.

**Characteristics:**
- Seeing patterns in random events
- Confusing correlation with causation
- Confirmation bias in pattern recognition
- Overinterpretation of coincidences

**Misinformation Cues:**
- Claims of causation from correlation
- Connecting unrelated events or phenomena
- "Studies show" without proper methodology
- Oversimplified cause-and-effect relationships
- Cherry-picked data or selective reporting

**Example Headlines:**
- "People who do X are 90% more likely to experience Y"
- "Scientists discover surprising link between A and B"
- "New study reveals hidden connection"
- "The real reason behind [common problem]"

## Label Combinations

Headlines often exhibit multiple psychological mechanisms simultaneously:

### Common Combinations (Based on our understanding):

1. **Peripheral + Naturalness:** "Doctors hate this natural miracle cure!"
2. **Availability + Illusory:** "Local outbreak linked to common household item"
3. **Central + Illusory:** "Research confirms correlation between X and Y"
4. **Peripheral + Availability:** "Celebrity's shocking health scare - could happen to you"

### Framework Interactions:

- **ELM Routes:** Headlines typically favor one route but may contain elements of both
- **Cognitive Biases:** Often work together to reinforce misinformation
- **Cross-Framework:** Peripheral route processing makes individuals more susceptible to cognitive biases


### Multi-Label Classification

Each headline can have multiple labels (0 or 1 for each mechanism):
- Labels are not mutually exclusive
- A headline can exhibit multiple biases simultaneously
- The model predicts probability for each label independently

### Threshold Selection

- Default threshold: 0.5
- Can ajdust this based on precision/recall requirements
- Different thresholds may be optimal for different labels

### Considerations

- Class imbalance may require weighted loss functions
- Some mechanisms may be more difficult to detect than others
- Context and domain knowledge important for accurate labeling