import unittest
from analyze import numeric,read_csv,stats,classify_operation


class ColdAnalysis(unittest.TestCase):
    def test_numeric(self):
        self.assertEqual(numeric('1,024.00'),1024)
        for value in ('nan','inf','-1'):
            with self.assertRaises(ValueError):
                numeric(value)
    def test_stats(self):
        self.assertEqual(stats([1,3,2])['mean'],2)
        self.assertEqual(stats([1,3,2])['p50'],2)
    def test_composite_attention_matmul(self):
        base=dict(operator='aten::matmul',roles=['attention'])
        self.assertEqual(classify_operation(dict(base,shapes=[[1,8,16,256],[1,8,256,984]])),('attention_qkt','linear'))
        self.assertEqual(classify_operation(dict(base,shapes=[[1,8,16,984],[1,8,984,256]])),('attention_sv','linear'))
    def test_rms_square(self):
        self.assertEqual(classify_operation(dict(operator='aten::square',roles=['rms_norm'],shapes=[[1,16,1024]])),('rms_pow','nonlinear_core'))
    def test_csv(self):
        from unittest.mock import Mock
        content='"ID","Kernel Name","Grid Size","Block Size","gpu__time_duration.sum","profiler__replayer_passes"\n'
        content+='"","","","","ns","pass"\n'
        content+='"0","op_00012/kernel","(2, 1, 1)","(256, 1, 1)","1,024.00","1.00"\n'
        rows=read_csv(Mock(read_text=lambda:content))
        self.assertEqual(rows[0]['op_id'],12)
        self.assertEqual(rows[0]['kernel_ns'],1024)
        self.assertEqual(rows[0]['grid'],[2,1,1])
    def test_library_nvtx_range_preserves_outer_owner(self):
        import csv,io
        from unittest.mock import Mock
        out=io.StringIO()
        writer=csv.writer(out,quoting=csv.QUOTE_ALL)
        writer.writerow(['ID','Kernel Name','Grid Size','Block Size','gpu__time_duration.sum','profiler__replayer_passes','thread Domain:Push/Pop_Range:Msg'])
        writer.writerow(['','','','','ns','pass',''])
        writer.writerow(['0','cub::Scan/kernel','(1, 1, 1)','(128, 1, 1)','2000','1','"<default domain>:cold_step:none" "<default domain>:op_00007:none" "cub::Scan:none"'])
        result=read_csv(Mock(read_text=lambda:out.getvalue()))[0]
        self.assertEqual(result['op_id'],7)
        self.assertEqual(result['kernel'],'kernel')


if __name__=='__main__':
    unittest.main()
