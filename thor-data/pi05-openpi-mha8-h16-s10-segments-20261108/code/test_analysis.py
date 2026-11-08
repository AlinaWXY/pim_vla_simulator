import unittest
from analyze import MODES,aggregate,stats


def case(suite='native_blocks',family='rope_qk',layer=0,value=.1):
    return dict(id=f'{suite}/{family}/{layer}',suite=suite,family=family,layer=layer,
        bitwise_equal=True,internal_evictions=0,evictions_per_cold_sample=1,
        measurements={m:dict(samples_ms=[value]*20,summary_ms=stats([value]*20)) for m in MODES})


class AggregationTests(unittest.TestCase):
    def test_layer_sums(self):
        rows=aggregate([case(layer=0,value=.1),case(layer=1,value=.2)])
        self.assertEqual(rows[0]['occurrences_per_step'],2)
        self.assertAlmostEqual(rows[0]['graph_entry_cold']['sum_case_means_ms'],.3)
        self.assertAlmostEqual(rows[0]['graph_entry_cold']['mean_occurrence_us'],150)
    def test_suites_never_merge(self):
        self.assertEqual(len(aggregate([case(),case(suite='handoff_composites')])),2)
    def test_internal_flush_rejected(self):
        c=case();c['internal_evictions']=1
        with self.assertRaises(ValueError):aggregate([c])
    def test_missing_sample_rejected(self):
        c=case();c['measurements']['graph_entry_cold']['samples_ms'].pop()
        with self.assertRaises(ValueError):aggregate([c])
    def test_nan_rejected(self):
        c=case();c['measurements']['graph_entry_cold']['samples_ms'][0]=float('nan')
        with self.assertRaises(ValueError):aggregate([c])
    def test_mismatched_output_rejected(self):
        c=case();c['bitwise_equal']=False
        with self.assertRaises(ValueError):aggregate([c])


if __name__=='__main__':unittest.main()
