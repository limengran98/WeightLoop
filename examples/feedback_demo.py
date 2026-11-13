"""Small synthetic walkthrough of fixed-program execution and native reload.

No dataset download, LLM call or training is required. This illustrates the
mechanics using synthetic observations.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse
import json
from pathlib import Path
import numpy as np
from weightloop import choose_update, functional_utility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve previous records")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(17)
    z = rng.normal(size=(225,128))
    hidden = rng.normal(0.,.03,128)
    # A deliberately planted task: two hidden features carry the outcome,
    # while the deployed output starts flat. This is a workflow illustration.
    truth = np.zeros(128)
    truth[7], truth[29] = .10, -.06
    w = np.zeros(128)
    bias = .1
    y = np.maximum(z,0.) @ truth + bias
    feedback, query = np.arange(75), np.arange(75,225)
    folds = np.arange(75)%3
    vector, selection = choose_update(z[feedback],y[feedback],w,bias,hidden,folds)
    np.savez(args.output/"original.npz",w=w,bias=np.array(bias),hidden_bias=hidden)
    np.savez(args.output/"updated.npz",w=vector[:128],bias=np.array(bias),hidden_bias=vector[128:])
    np.savez(args.output/"feedback.npz",z=z[feedback],y=y[feedback],folds=folds)
    # The same native forward consumes saved model tensors, with no reranker.
    def forward(ww, bb):return np.maximum(z[query]+bb-hidden,0.) @ ww+bias
    before = forward(w,hidden)
    after = forward(vector[:128],vector[128:])
    with np.load(args.output/"updated.npz",allow_pickle=False) as loaded:
        replay = forward(loaded["w"],loaded["hidden_bias"])
    assert np.array_equal(after,replay)
    report = {"example":"synthetic-mechanics-only", "selection":selection,
              "feedback_count":75,"query_count":len(query),"llm_calls":0,"backpropagation_steps":0,
              "native_reload_max_error":float(np.max(np.abs(after-replay))),
              "query_MSE":{"original":float(np.mean((before-y[query])**2)),"updated":float(np.mean((after-y[query])**2))},
              "query_BU20":{"original":functional_utility(before,y[query],20),"updated":functional_utility(after,y[query],20)}}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":selection["status"],"recipe":selection["recipe"],"changed_parameters":selection["changed_parameters"],"native_reload_max_error":report["native_reload_max_error"],"report":str(args.output/"report.json")},indent=2))


if __name__=="__main__":main()
