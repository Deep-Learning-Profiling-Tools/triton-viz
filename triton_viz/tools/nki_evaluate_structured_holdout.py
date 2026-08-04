"""Audit zero-operator-fit structured Level-A against mapped holdout artifacts."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from triton_viz.tools.nki_cost_model import ComputeCalibration, StructuredControlCalibration
from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature

FIELDS=["case","region","family","engine","actual_instruction_count","predicted_instruction_count",
        "count_error_pct","actual_active_ns","predicted_active_ns","active_error_pct"]

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path)
    p.add_argument("--compute-calibration-csv",type=Path,required=True);p.add_argument("--structured-control-csv",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args(argv)
    level_b=ComputeCalibration.from_csv(a.compute_calibration_csv);model=StructuredControlCalibration.from_csv(a.structured_control_csv);out=[]
    excluded={"DRAIN","NOTIFY","EVENT_SEMAPHORE","EVENT_SEMAPHORE_RANGE_CLEAR","SET_ORDERING_MODE"}
    for trace in sorted(a.root.glob("*/trace.jsonl")):
        case=trace.parent; ap=case/"hardware/source_mapping/audit.json";mp=case/"hardware/source_mapping/instruction_mapping.csv"
        if not ap.is_file() or not mp.is_file():continue
        events=[json.loads(x) for x in trace.read_text().splitlines() if x.strip()];_annotate_fusion_signature(events); groups={}
        for e in events:
            if e.get("region_ir") is not None:groups.setdefault(int(e["fusion_group"]),e["region_ir"])
        audit=json.loads(ap.read_text()); mappings=list(csv.DictReader(mp.open()))
        for group,ir in groups.items():
            points=model.predict_points(ir)
            from triton_viz.tools.nki_region_ir import structural_family
            for engine in ("vector","scalar"):
                actual=[r for r in mappings if r["engine"]==engine and r["fusion_group"]==str(group) and r["opcode"] not in excluded]
                predicted=points.get(engine,(0,0,0)); streams=2 if engine=="vector" else 1
                active=float(audit["engines"][engine]["regions"].get(str(group),0)); pred_active=predicted[0]*level_b.instruction_ns(engine,ir["dtype"],streams,int(ir["free_dim"]))
                row={"case":case.name,"region":group,"family":structural_family(ir),"engine":engine,
                     "actual_instruction_count":len(actual),"predicted_instruction_count":predicted[1],
                     "actual_active_ns":active,"predicted_active_ns":pred_active}
                row["count_error_pct"]=abs(predicted[1]-len(actual))/max(1,len(actual))*100
                row["active_error_pct"]=abs(pred_active-active)/max(1,active)*100;out.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(out)
    for engine in ("vector","scalar"):
        rows=[r for r in out if r["engine"]==engine]
        if rows:print(engine,"count MAPE",sum(r["count_error_pct"] for r in rows)/len(rows),"active MAPE",sum(r["active_error_pct"] for r in rows)/len(rows))
    return 0
if __name__=="__main__":raise SystemExit(main())
