import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_evaluation import fit,gaps,merge

class KernelFit(unittest.TestCase):
    def test_never_add_disjoint_gaps(self):
        ws=[dict(begin_us=0,end_us=3,duration_us=3),dict(begin_us=5,end_us=8,duration_us=3)]
        self.assertEqual(fit(ws,[dict(us=4,launch_us=0)])['placed_kernels'],0)
    def test_no_skip_dependency(self):
        ws=[dict(begin_us=0,end_us=3,duration_us=3)]
        self.assertEqual(fit(ws,[dict(us=4,launch_us=0),dict(us=1,launch_us=0)])['placed_kernels'],0)
    def test_launch_reservation_and_exact_edge(self):
        ws=[dict(begin_us=0,end_us=4,duration_us=4)]
        r=fit(ws,[dict(us=3,launch_us=1),dict(us=.1,launch_us=0)],launch=True)
        self.assertEqual(r['placed_kernels'],1);self.assertEqual(r['overlap_ms'],.003)
        self.assertEqual(r['placements'][0]['end_us'],4)
    def test_shared_resource_union(self):
        self.assertEqual(gaps(merge([(1,3),(2,4),(6,7)]),0,9),[[0,1],[4,6],[7,9]])

if __name__=='__main__':unittest.main()
