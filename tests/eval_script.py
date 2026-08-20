from datasets import Dataset
from ragas import evaluate
# FIX: Import the precise collection paths for Ragas v0.4+
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

# FIX: Update keys to modern standard names ('user_input' and 'reference')
data = {
    'user_input': ["What technique of decoupling has been used in the project named bluprnt"],
    'response': ["The decoupling technique which has been used is Producer-Consumer technique"],
    'contexts': [["Projects section Bluprnt project"]],
    'reference': ["Producer Consumer"]
}

datasets = Dataset.from_dict(data)

# FIX: Instantiate the exact classes
results = evaluate(
    dataset=datasets,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()]
)

print(results)
