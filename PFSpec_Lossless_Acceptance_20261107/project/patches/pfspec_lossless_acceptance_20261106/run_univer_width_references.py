"""Complete two predeclared width experiments without overlapping GPU work."""
import json,subprocess,time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106')
PY='/root/miniconda3/envs/gaussiangpt/bin/python'
SOURCE=ROOT/'source'
OUT=ROOT/'paper08_univer'


def run(name,*arguments):
    with (OUT/f'{name}.log').open('w') as stream:
        return subprocess.call([PY,*map(str,arguments)],stdout=stream,stderr=subprocess.STDOUT)


def main():
    original=ROOT/'paper02_draft_opd/development_online_01'
    original_rows=[json.loads(line) for line in (original/'records.jsonl').read_text().splitlines() if line.strip()]
    selected_rows=[next(row for row in original_rows if row['method']=='w1' and row['cohort']==cohort) for cohort in ['bos','completion']]
    original_inputs=json.loads((ROOT/'paper08_univer/development_online_01/plan.json').read_text())['contexts']
    wanted={r['pair_id'] for r in selected_rows}
    validation_inputs=[r for r in original_inputs if r['pair_id'] in wanted]
    validation=OUT/'common_adapter_validation'
    validation.mkdir()
    (validation/'tokens').symlink_to(original/'tokens',target_is_directory=True)
    (validation/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected_rows))
    (validation/'inputs.json').write_text(json.dumps(dict(contexts=validation_inputs),indent=2)+'\n')
    (validation/'selected_development_model.json').write_text(json.dumps(dict(status='complete',method='diffusion_full_w64',checkpoint=str(ROOT/'paper06_dflash/training/diffusion_context_w64/selected.pt'),purpose='Validate the development adapter; this file does not select a formal model'),indent=2)+'\n')
    validation_code=run('common_adapter_validation',SOURCE/'offline_final_common.py','--selection',validation/'selected_development_model.json','--records',validation/'records.jsonl','--inputs',validation/'inputs.json','--out',validation/'results','--development-validation','--rows-per-request',4)
    (validation/'execution.json').write_text(json.dumps(dict(returncode=validation_code,status='complete' if validation_code==0 else 'failed'),indent=2)+'\n')
    results=[]
    for width in [32,64]:
        item=dict(width=width,status='running',started_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()))
        numeric=OUT/f'width_{width}_numeric.json'
        code=run(f'width_{width}_numeric',SOURCE/'probe_univer_width_numeric.py','--width',width,'--out',numeric)
        if code:
            item.update(status='failed',stage='numeric execution',returncode=code)
        else:
            data=json.loads(numeric.read_text())
            probabilities=[v for row in data['rows'] for v in row['fresh_tree_vs_linear']['tree162']]
            valid=all(v['support_equal'] and v['tv']<=1e-5 for v in probabilities)
            item.update(maximum_policy_tv=max(v['tv'] for v in probabilities),numeric_valid=valid)
            if not valid:item.update(status='invalid',stage='Target policy comparison')
            else:
                stages=[('native',SOURCE/'validate_univer_branches.py','--width',width,'--eager-only','--out',OUT/f'width_{width}_native.json'),
                    ('prepare',SOURCE/'prepare_univer.py','--width',width),
                    ('validation',SOURCE/'evaluate_univer.py','--root',OUT,'--methods',f'diffusion_univer_w{width}','--capacity',32768,'--inputs','/root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json','--expected-gpu',5090,'--validate','--out',OUT/f'development_validation_w{width}'),
                    ('natural',SOURCE/'evaluate_univer.py','--root',OUT,'--methods',f'diffusion_univer_w{width}','--capacity',32768,'--inputs','/root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json','--expected-gpu',5090,'--out',OUT/f'development_online_w{width}'),
                    ('common',SOURCE/'offline_univer.py','--width',width,'--out',OUT/f'development_curve_w{width}')]
                for name,*arguments in stages:
                    code=run(f'width_{width}_{name}',*arguments)
                    if code:
                        item.update(status='failed',stage=name,returncode=code)
                        break
                else:item.update(status='complete')
        item['ended_utc']=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())
        results.append(item)
        (OUT/'width_reference_results.json').write_text(json.dumps(dict(status='running',results=results),indent=2)+'\n')
        print(json.dumps(item),flush=True)
    (OUT/'width_reference_results.json').write_text(json.dumps(dict(status='complete',results=results),indent=2)+'\n')


if __name__=='__main__':main()
