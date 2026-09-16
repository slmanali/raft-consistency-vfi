import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from raft_consistency_vfi.cli import main
from raft_consistency_vfi.data import read_manifest, save_image


class WorkflowTests(unittest.TestCase):
    def fixture(self,root):
        for name,value in (("a",0.2),("target",0.3),("b",0.4)):
            save_image(root/f"{name}.png",np.full((16,24,3),value))
        manifest=root/"manifest.csv"
        manifest.write_text("id,input0,target,input1,split,group\ncase_1,a.png,target.png,b.png,easy,sequence_1\n")
        return manifest

    def test_average_evaluation_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.fixture(root); out=root/"result"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["evaluate","--method","average","--manifest",str(manifest),"--output",str(out)])
            meta=json.loads((out/"run.json").read_text())
            self.assertEqual(meta["status"],"completed")
            with (out/"per_image.csv").open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1)
            self.assertGreater(float(rows[0]["psnr"]),40)
            self.assertFalse((out/"images").exists())

    def test_duplicate_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.fixture(root)
            with manifest.open("a") as f: f.write("case_1,a.png,target.png,b.png,hard,sequence_1\n")
            with self.assertRaises(ValueError): read_manifest(manifest)

    def test_ground_truth_cannot_equal_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.fixture(root)
            manifest.write_text(manifest.read_text().replace("a.png,target.png,b.png","a.png,a.png,b.png"))
            with self.assertRaises(ValueError): read_manifest(manifest)

    def test_external_float_prediction_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.fixture(root)
            predictions=root/"pred"; predictions.mkdir()
            np.save(predictions/"case_1.npy",np.full((16,24,3),0.3,np.float32))
            provenance=root/"provenance.json"
            provenance.write_text(json.dumps({"model":"test_fixture","weights_sha256":"unit-test","code_commit":"unit-test"}))
            main(["score","--manifest",str(manifest),"--predictions",str(predictions),
                "--provenance",str(provenance),"--output",str(root/"scored")])
            self.assertEqual(json.loads((root/"scored"/"run.json").read_text())["status"],"completed")

    def test_snu_manifest_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/"frames"; seq=data/"test"/"seq"; seq.mkdir(parents=True)
            for i in (0,1,2): save_image(seq/f"{i}.png",np.zeros((12,12,3)))
            lists=root/"lists"; lists.mkdir()
            for mode in ("easy","medium","hard","extreme"):
                (lists/f"test-{mode}.txt").write_text("data/SNU-FILM/test/seq/0.png data/SNU-FILM/test/seq/1.png data/SNU-FILM/test/seq/2.png\n")
            with contextlib.redirect_stdout(io.StringIO()):
                main(["make-snu-manifest","--data-root",str(data),"--list-root",str(lists),"--output",str(root/"snu.csv")])
            self.assertEqual(len(read_manifest(root/"snu.csv")),4)


if __name__=="__main__": unittest.main()
