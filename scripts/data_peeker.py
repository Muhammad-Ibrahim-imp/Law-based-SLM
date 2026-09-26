from datasets import load_dataset

# Try the Atticus contracts subset first (real, formal legal contracts)
ds = load_dataset(
    "pile-of-law/pile-of-law",
    "courtlistener_opinions",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

for example in ds.take(1):
    print(example)