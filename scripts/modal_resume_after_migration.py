import os
import time
import modal

app = modal.App('law-slm-resume-controller')

@app.function(cpu=0.25, memory=256, timeout=7200)
def resume_after_migration(migration_call: str, sft_only: bool = False):
    result = modal.FunctionCall.from_id(migration_call).get(timeout=6900)
    if not result or not result.get('files'):
        raise ValueError('migration did not return a verified file inventory')
    print('Migration complete; running CPU preflight', result, flush=True)
    modal.Function.from_name('law-slm-legal-finetuning', 'preflight').remote(sft_only=sft_only)
    train = modal.Function.from_name('law-slm-legal-finetuning', 'train_domain')
    calls = {domain: train.spawn(domain, sft_only=sft_only).object_id for domain in (('criminal',) if sft_only else ('family','criminal'))}
    print('H200 training calls:', calls, flush=True)
    return calls

@app.local_entrypoint()
def main(migration_call: str, sft_only: bool = False):
    call = resume_after_migration.spawn(migration_call, sft_only)
    print('Controller call:', call.object_id, flush=True)
    time.sleep(3)
    os._exit(0)
