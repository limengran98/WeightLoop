"""Archived fixed update language and matched numerical solvers.

No files, API calls or target acquisition. Programs receive training-fold arrays
only. New tasks share this implementation; task-specific developer changes must
be attributed separately from proposals made by a recorded runtime LLM.
"""
from __future__ import annotations

import ast
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

VARIABLES = ('residual', 'target', 'prediction', 'age', 'condition')
FUNCTIONS = {'abs': np.abs, 'sqrt': lambda x: np.sqrt(np.maximum(x, 0)),
             'log1p': lambda x: np.log1p(np.maximum(x, 0)),
             'exp': lambda x: np.exp(np.clip(x, -20, 20)), 'tanh': np.tanh,
             'minimum': np.minimum, 'maximum': np.maximum, 'clip': np.clip}
KEYS = {'loss', 'ridge', 'preservation', 'step', 'scope', 'condition', 'weight_program'}


def parse_program(expression):
    if not isinstance(expression, str) or not 1 <= len(expression) <= 320:
        raise ValueError('Weight program must be a bounded expression')
    tree = ast.parse(expression, mode='eval')
    nodes = list(ast.walk(tree))
    if len(nodes) > 72:
        raise ValueError('Weight program exceeds 72 AST nodes')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
               ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.USub, ast.UAdd)
    for node in nodes:
        if not isinstance(node, allowed):
            raise ValueError('Unsupported program operation: ' + type(node).__name__)
        if isinstance(node, ast.Name) and node.id not in (*VARIABLES, *FUNCTIONS):
            raise ValueError('Unknown program variable: ' + node.id)
        if isinstance(node, ast.Name) and node.id in FUNCTIONS and not any(
                isinstance(parent, ast.Call) and parent.func is node for parent in nodes):
            raise ValueError('Numerical function names require a call')
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float)
                or not np.isfinite(node.value) or abs(node.value) > 1000):
            raise ValueError('Constants must be finite bounded numbers')
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise ValueError('Only registered numerical functions are callable')
            arity = 3 if node.func.id == 'clip' else 2 if node.func.id in ('minimum', 'maximum') else 1
            if len(node.args) != arity:
                raise ValueError('Incorrect function argument count')
    return tree


def run_program(expression, variables, shape):
    """Interpret the restricted AST without eval, imports, attributes or I/O."""
    def walk(node):
        if isinstance(node, ast.Expression): return walk(node.body)
        if isinstance(node, ast.Constant): return float(node.value)
        if isinstance(node, ast.Name): return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            x = walk(node.operand)
            return -x if isinstance(node.op, ast.USub) else x
        if isinstance(node, ast.Call): return FUNCTIONS[node.func.id](*[walk(a) for a in node.args])
        a, b = walk(node.left), walk(node.right)
        if isinstance(node.op, ast.Add): return a+b
        if isinstance(node.op, ast.Sub): return a-b
        if isinstance(node.op, ast.Mult): return a*b
        # Division is total, with an explicit fixed signed denominator floor.
        b = np.asarray(b)
        return a / np.where(np.abs(b) < 1e-6, np.where(b < 0, -1e-6, 1e-6), b)
    with np.errstate(all='raise'):
        value = np.broadcast_to(walk(parse_program(expression)), shape).copy()
    if not np.isfinite(value).all(): raise ValueError('Nonfinite program output')
    return np.clip(value, .05, 20.)


def validate_spec(spec, scopes, conditions, binary=False):
    if not isinstance(spec, dict) or set(spec) != KEYS:
        raise ValueError('Experiment requires exactly ' + ','.join(sorted(KEYS)))
    choices = ('bce', 'brier') if binary else ('mse', 'smooth_mae', 'huber')
    if spec['loss'] not in choices: raise ValueError('Loss incompatible with native output')
    for key, lo, hi in [('ridge', 1e-7, 1e3), ('preservation', 0., 100.), ('step', .05, 1.5)]:
        if type(spec[key]) not in (int, float) or not np.isfinite(spec[key]) or not lo <= spec[key] <= hi:
            raise ValueError('Invalid bounded ' + key)
    if spec['scope'] not in scopes or spec['condition'] not in conditions:
        raise ValueError('Unregistered scope/condition')
    parse_program(spec['weight_program'])
    return dict(spec)


def specification(loss='mse', ridge=.01, preservation=0., step=1., scope='full',
                  condition='all', weight_program='1'):
    return dict(loss=loss, ridge=ridge, preservation=preservation, step=step,
                scope=scope, condition=condition, weight_program=weight_program)


def anchors(binary=False):
    loss = 'bce' if binary else 'mse'
    return [specification(loss, r) for r in (.0001, .01, 1.)] + [
        specification('bce' if binary else 'smooth_mae', .01,
                      weight_program='1 + 9 * target' if binary else '1')]


def fixed_extra(binary=False):
    loss = 'bce' if binary else 'mse'
    return [specification(loss, r, p) for r in (.00001, .001, .1, 10.) for p in (0., .1)]




def fit(p0, jacobian, w0, scales, y, reference_jacobian, indices, spec, *,
        output_scale, age=None, condition=None, optimizer='lbfgs', max_iterations=80):
    """Fit an exact affine native head, preserving the total-patch W0 anchor.

    Uses uniform entity mass, output-mean loss, and full-pool ridge scaling.
    The program is fitted/evaluated on this training fold only. No RMS clipping
    or inherited Depth drift threshold; preservation is a registered soft cost.
    """
    started=time.perf_counter()
    p0,y=np.asarray(p0,dtype=float),np.asarray(y,dtype=float)
    w0,scales=np.asarray(w0,dtype=float),np.asarray(scales,dtype=float)
    jac=np.asarray(jacobian,dtype=float);ref=np.asarray(reference_jacobian,dtype=float)
    if p0.ndim!=2 or min(p0.shape)<1 or w0.ndim!=1 or scales.shape!=w0.shape:
        raise ValueError('Invalid native head array ranks')
    n,d=p0.shape;ix=np.asarray(indices)
    if ix.ndim!=1 or not ix.size or not np.issubdtype(ix.dtype,np.integer) or np.any(ix<0) or np.any(ix>=len(w0)) or len(np.unique(ix))!=len(ix):
        raise ValueError('Scope must contain unique in-range integer indices')
    if ref.ndim!=3 or ref.shape[1:]!=(d,len(w0)) or not len(ref):
        raise ValueError('Invalid reference shape')
    if any(not np.isfinite(x).all() for x in (p0,w0,scales,jac,ref)) or np.any(scales<=0):
        raise ValueError('Native arrays must be finite and scales positive')
    if y.shape!=p0.shape or jac.shape!=(n,d,len(w0)) or not np.isfinite(y).all():
        raise ValueError('Invalid registered affine observations')
    if not np.isfinite(output_scale) or output_scale<=0:raise ValueError('Invalid base scale')
    binary=spec['loss'] in ('bce','brier')
    scale=1. if binary else output_scale
    full=(jac*scales/scale).reshape(-1,len(w0))
    a=full[:,ix];r=(ref*scales/scale).reshape(-1,len(w0))[:,ix]
    base=(p0/scale).ravel();target=(y/scale).ravel()
    if binary and (np.any(y<0) or np.any(y>1)):raise ValueError('BCE labels outside [0,1]')
    original_residual=(expit(p0)-y) if binary else (p0-y)/output_scale
    variables=dict(residual=original_residual,target=y if binary else y/output_scale,
                   prediction=expit(p0) if binary else p0/output_scale,
                   age=np.broadcast_to(np.asarray(np.zeros(n) if age is None else age)[:,None],(n,d)),
                   condition=np.broadcast_to(np.asarray(np.zeros(n) if condition is None else condition)[:,None],(n,d)))
    raw_weights=run_program(spec['weight_program'],variables,(n,d))
    q=(raw_weights/raw_weights.sum()).ravel()
    beta=spec['ridge']*max(float(np.mean(np.sum(full**2,axis=0))/len(full)),1e-12)
    refgram=(r.T@r)/max(len(r),1)
    penalty=spec['preservation']*refgram+beta*np.eye(len(ix))
    cost=dict(objective_evaluations=0,optimizer_steps=0,linear_solves=0)
    def fun(z):
        cost['objective_evaluations']+=1
        pred=base+a@z;e=pred-target
        if spec['loss']=='bce':
            values=np.logaddexp(0,pred)-target*pred;g=expit(pred)-target
        elif spec['loss']=='brier':
            probability=expit(pred);error=probability-target
            values=error**2;g=2*error*probability*(1-probability)
        elif spec['loss']=='smooth_mae':
            v=np.hypot(e,.01);values=v-.01;g=e/v
        elif spec['loss']=='huber':
            ae=np.abs(e);values=np.where(ae<=1,.5*e**2,ae-.5);g=np.clip(e,-1,1)
        else:values=e**2;g=2*e
        return float(q@values+z@penalty@z), a.T@(q*g)+2*penalty@z
    initial=fun(np.zeros(len(ix)))[0]
    if spec['loss']=='mse' and optimizer!='adam':
        z=np.linalg.solve(a.T@(q[:,None]*a)+penalty,-a.T@(q*(base-target)))
        cost['linear_solves']=1;cost['optimizer_steps']=1;status='analytic'
    elif optimizer=='adam':
        z=np.zeros(len(ix));m=z.copy();v=z.copy();best=z.copy();best_loss=initial
        # Shared normalized-coordinate Adam recipe, no task-specific LR tuning.
        for t in range(1,max_iterations+1):
            value,g=fun(z)
            if value<best_loss:best_loss,best=value,z.copy()
            m=.9*m+.1*g;v=.999*v+.001*g*g
            z-=.03*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        if fun(z)[0]<best_loss:best=z.copy()
        z=best;cost['optimizer_steps']=max_iterations;status='fixed_adam_budget'
    else:
        solved=minimize(fun,np.zeros(len(ix)),method='L-BFGS-B',jac=True,
                        options={'maxiter':max_iterations,'ftol':1e-10,'gtol':1e-7,'maxls':20})
        z=solved.x;cost['optimizer_steps']=int(solved.nit);status=str(solved.message)
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite fitted patch')
    optimized=fun(z)[0];z=z*spec['step']
    result=w0.copy();result[ix]+=scales[ix]*z
    result[ix]=result[ix].astype(np.float32).astype(np.float64)
    final_z=(result[ix]-w0[ix])/scales[ix]
    final=fun(final_z)[0]
    return result,dict(status=status,objective_before=initial,objective_optimized=optimized,
        objective_after_native_rounding=final,fixed_beta=beta,weight_min=float(raw_weights.min()),
        weight_max=float(raw_weights.max()),native_changed_parameters=int(np.count_nonzero(result.astype(np.float32)!=w0.astype(np.float32))),
        total_patch_scaled_rms=float(np.sqrt(np.mean(((result-w0)/scales)**2))),
        cost={**cost,'seconds':time.perf_counter()-started},optimizer=optimizer,
        program_training_fold_only=True,actual_native_dtype='float32')
