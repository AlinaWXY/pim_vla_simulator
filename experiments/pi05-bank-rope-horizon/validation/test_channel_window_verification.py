import unittest
from verify_channel_windows import classify_phase, frontend_tick


class ChannelWindowVerificationTests(unittest.TestCase):
    def test_native_clock_edge_is_not_shifted_into_next_phase(self):
        self.assertEqual(frontend_tick(10, 32), 289)
        timelines = [[dict(clock='10',command='WRGB')]]+[[] for _ in range(15)]
        self.assertEqual(classify_phase(timelines, 257, 290, 32), ([1]+[0]*15, False))
        self.assertEqual(classify_phase(timelines, 290, 321, 32), ([0]*16, True))

    def test_every_channel_must_have_zero_external_commands(self):
        timelines = [[dict(clock='10',command='MACAB')] for _ in range(16)]
        self.assertTrue(classify_phase(timelines, 257, 321, 32)[1])
        timelines[15].append(dict(clock='10',command='RDRES'))
        counts, zero = classify_phase(timelines, 257, 321, 32)
        self.assertFalse(zero)
        self.assertEqual(counts, [0]*15+[1])


if __name__ == '__main__':
    unittest.main()
