"""Verify the current core results and three-training-seed summaries.

Pure standard-library replay of released source tables; no scientific fits.
"""
import csv,hashlib,json,math
from pathlib import Path
from statistics import fmean,stdev
ROOT=Path(__file__).resolve().parents[1]
def rows(name):
 with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def check(role,summary,keys,metric,mean_key='mean',sd_key='std',seed_key='seed'):
 raw=rows(role);published=rows(summary);assert raw and published,(role,'empty table')
 grouped={}
 for r in raw:grouped.setdefault((tuple(r[k] for k in keys),r[seed_key]),[]).append(float(r[metric]))
 values={}
 for (key,seed),v in grouped.items():values.setdefault(key,[]).append(fmean(v))
 expected={tuple(r[k] for k in keys):r for r in published};assert values.keys()==expected.keys(),role
 maximum=0.
 for key,v in values.items():
  assert len(v)==3,(role,key,'exactly three training seeds required')
  errs=[abs(fmean(v)-float(expected[key][mean_key])),abs(stdev(v)-float(expected[key][sd_key]))];maximum=max(maximum,*errs)
  assert max(errs)<1e-12,(role,key,errs)
 return dict(table=summary,metric=metric,groups=len(values),max_absolute_error=maximum)
def primary():
 data=rows('results/static_cases/final_metrics.tsv');assert len(data)==8
 tasks={'BIO03','SEIS03','MET02','PHY05'};assert {r['task'] for r in data}==tasks
 assert {r['provider_condition'] for r in data}=={'deepseek'}
 effects=[]
 for task in sorted(tasks):
  for condition in ['deepseek']:
   r=[x for x in data if x['task']==task and x['provider_condition']==condition];assert len(r)==2
   methods={x['method']:x for x in r};assert set(methods)=={'Fixed','Adam80'}
   f,a=methods['Fixed'],methods['Adam80']
   for k in ['archive_seed','feedback_entities','evaluation_entities','allowed_native_parameters','metric','unit']:assert f[k]==a[k]
   assert f['archive_seed']=='17'
   fv,av=float(f['native_error']),float(a['native_error']);ratio=fv/av;gain=100*(av-fv)/av
   assert math.isfinite(ratio) and 0<ratio<1
   assert abs(ratio-float(f['error_relative_to_Adam80']))<1e-12
   assert abs(gain-float(f['error_reduction_vs_Adam80_percent']))<1e-10
   assert float(a['error_relative_to_Adam80'])==1.
   effects.append(gain)
 parity=json.loads((ROOT/'results/static_cases/feedback_parity.json').read_text());assert len(parity['cases'])==4
 for c in parity['cases']:
  assert c['evaluation_ids_excluded_from_feedback']
  assert set(c['providers'])=={'deepseek'}
  assert all(p['ordered_feedback_ids_and_targets_identical'] for p in c['providers'].values())
  assert len(c['ordered_feedback_ids_sha256'])==len(c['target_array_sha256'])==64
 protocols=json.loads((ROOT/'results/static_cases/case_protocols.json').read_text())['cases'];assert len(protocols)==4
 assert {r['task'] for r in protocols}==tasks and {r['provider_condition'] for r in protocols}=={'deepseek'}
 updates=rows('results/static_cases/native_updates.tsv');assert len(updates)==12
 assert {r['provider_condition'] for r in updates}=={'deepseek'}
 for task in tasks:
  for k in range(1,4):
   rr=[r for r in updates if r['task']==task and int(r['update'])==k];assert len(rr)==1
   assert len({r['changed_from_original'] for r in rr})==1
   for r in rr:
    assert 0<=int(r['changed_from_original'])<=int(r['allowed_native_parameters'])
    if r['status']=='RETAIN_CURRENT':assert int(r['changed_from_previous'])==0
 stream=rows('results/static_cases/feedback_stream.tsv');assert len(stream)==24
 assert {r['provider_condition'] for r in stream}=={'deepseek'}
 for task in tasks:
  protocol=next(r for r in protocols if r['task']==task);sizes=protocol['role_sizes']
  prefixes=[sizes['F0'],sizes['F0']+sizes['B1'],sizes['F0']+sizes['B1']+sizes['B2']]
  for k,role in [(1,'B1'),(2,'B2'),(3,'E')]:
   rr=[r for r in stream if r['task']==task and int(r['update'])==k];assert len(rr)==2
   methods={r['method']:r for r in rr};assert set(methods)=={'Fixed','Adam80'}
   assert {r['predicted_role'] for r in rr}=={role}
   assert all(int(r['feedback_entities'])==prefixes[k-1] for r in rr)
   fixed,adam=methods['Fixed'],methods['Adam80']
   assert abs(float(fixed['native_error'])/float(adam['native_error'])-float(fixed['error_relative_to_Adam80']))<1e-12
   if k==3:
    for method,row in methods.items():
     final=next(r for r in data if r['task']==task and r['method']==method)
     assert math.isclose(float(row['native_error']),float(final['native_error']),rel_tol=1e-12,abs_tol=1e-15)
 return dict(primary_comparisons=len(effects),final_rows=len(data),range_percent=[min(effects),max(effects)],feedback_parity='PASS',native_updates=len(updates),feedback_stream_rows=len(stream),protocol_cases=len(protocols))
def rna_calibration():
 prefix='results/rna_calibration/'
 split=rows(prefix+'split_metrics.tsv');seed=rows(prefix+'seed_metrics.tsv');summary=rows(prefix+'summary.tsv')
 assert (len(split),len(seed),len(summary))==(54,18,6)
 models={'scGPT_native','scFoundation_GEARS_native'};arms={'original','block_adapt','weightloop'};seeds={'20261109','20261110','20261111'}
 for data in [split,seed,summary]:
  assert {r['version'] for r in data}=={'v3'}
  assert {r['model'] for r in data}==models and {r['arm'] for r in data}==arms
  assert all(math.isfinite(float(r['MSE'] if 'MSE' in r else r['MSE_mean'])) for r in data)
 assert {r['root_training_seed'] for r in split}=={r['root_training_seed'] for r in seed}==seeds
 split_groups={};seed_values={}
 for row in split:
  key=tuple(row[k] for k in ['model','arm','root_training_seed'])
  split_groups.setdefault(key,[]).append(row)
  assert int(row['targets'])==20
 for row in seed:
  key=tuple(row[k] for k in ['model','arm','root_training_seed']);assert key not in seed_values
  assert int(row['split_count'])==3 and int(row['target_instances'])==60
  seed_values[key]=float(row['MSE'])
 assert split_groups.keys()==seed_values.keys() and len(split_groups)==18
 maximum=0.
 for key,group in split_groups.items():
  assert len(group)==3 and {r['split_index'] for r in group}=={'0','1','2'}
  error=abs(fmean(float(r['MSE']) for r in group)-seed_values[key]);maximum=max(maximum,error)
  assert error<1e-14,(key,error)
 checks=[check(prefix+'split_metrics.tsv',prefix+'summary.tsv',['model','arm'],'MSE','MSE_mean','MSE_std','root_training_seed'),check(prefix+'seed_metrics.tsv',prefix+'summary.tsv',['model','arm'],'MSE','MSE_mean','MSE_std','root_training_seed')]
 for model in models:
  method={r['arm']:r for r in summary if r['model']==model};assert set(method)==arms
  original=float(method['original']['MSE_mean'])
  for row in method.values():
   assert int(row['training_seed_count'])==int(row['data_split_count_per_seed'])==3
   gain=100*(original-float(row['MSE_mean']))/original
   assert abs(gain-float(row['relative_MSE_reduction_percent']))<1e-10
 protocol=json.loads((ROOT/prefix/'protocol.json').read_text())
 assert protocol['roles']==dict(base_adaptation_targets=40,feedback_targets=16,query_targets=20)
 assert protocol['root_training_seeds']==[20261109,20261110,20261111] and protocol['split_indices']==[0,1,2]
 shared=protocol['shared_adaptation_conditions']
 assert all(shared[k] for k in ['same_initial_BASE','same_16_feedback_targets','same_two_fold_feedback_validation','same_complete_v3_native_parameter_allowlist','no_op_available','selected_settings_refit_using_all_feedback'])
 assert not shared['query_response_access_during_editing'] and shared['inner_fold_target_counts']==[8,8]
 assert set(protocol['methods'])==arms and protocol['original_method_ids_preserved']
 return dict(split_rows=len(split),seed_means=len(seed),summary_groups=len(summary),split_to_seed_max_absolute_error=maximum,three_seed_checks=checks)
def manifest():
 document=json.loads((ROOT/'results/manifest.json').read_text());files=document['files']
 assert len({entry['path'] for entry in files})==len(files)
 released={str(path.relative_to(ROOT)) for path in (ROOT/'results').rglob('*') if path.is_file() and path.name!='manifest.json'}
 assert released=={entry['path'] for entry in files}
 for entry in files:
  path=ROOT/entry['path'];content=path.read_bytes()
  assert len(content)==entry['bytes'] and hashlib.sha256(content).hexdigest()==entry['sha256'],entry['path']
 return dict(files=len(files),sha256='PASS')
def main():
 checks=[check('results/additional_functional/role_metrics.tsv','results/additional_functional/summary.tsv',['task','arm','budget'],'BU'),check('results/additional_sequential/discovery_per_seed.tsv','results/additional_sequential/discovery_summary.tsv',['construction','case','round'],'normalized_discovery','mean','sample_SD')]
 print(json.dumps(dict(status='PASS_DISPLAYED_RESULT_REPLAY',primary=primary(),original_three_seed_summaries=checks,rna_calibration=rna_calibration(),manifest=manifest()),indent=2))
if __name__=='__main__':main()
