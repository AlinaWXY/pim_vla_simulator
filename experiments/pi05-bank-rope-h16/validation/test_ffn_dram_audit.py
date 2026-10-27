"""Counterexamples for actual Z array/port ordering and bounded input state."""
from collections import Counter
import copy
import unittest

import audit_ffn_dram as A


def native(command='MACAB', clock=1, byte=A.DOWN_BASE, **changes):
    row, within = divmod(byte, 2048)
    e = dict(command=command, clock=clock, Channel=0, Rank=1, BankGroup=-1,
             Bank=-1, Row=row, Column=within//32, pim_buffer=0, pim_rank_mask=2)
    e.update(changes)
    return e


def rich(operation, clock, ready_at, bank=0, column=0, buffer=-1, **changes):
    e = dict(operation=operation, clock=clock, ready_at=ready_at, Channel=0, Rank=1,
             BankGroup=bank//4, Bank=bank % 4, Row=176, Column=column, buffer=buffer,
             bytes=128 if operation == 'PIM_Z_GATHER' else 32,
             service_cycles=ready_at-clock, generation=0, bank_mask=1 << bank, mac_columns=0)
    e.update(changes)
    return e


def streamed_stores():
    # The next feature may gather while the previous bank is in write recovery.
    gathers = [rich('PIM_Z_GATHER', 100+10*b, 108+10*b, b) for b in range(2)]
    returns = [rich('PIM_Z_RETURN', 108+10*b, 110+10*b, b) for b in range(2)]
    stores = [rich('PIM_Z_STORE', 110+10*b, 151+10*b, b) for b in range(2)]
    return gathers, returns, stores


def ports(gathers=(), returns=(), loads=(), stores=(), reads=(), **kw):
    return A.audit_stream_ports(list(gathers), list(returns), list(loads), list(stores), list(reads),
        local_ck=2, read_ck=25, write_ck=41, write_consume_ck=4, **kw)


def fragment_sample(count=3, slots=2):
    loads, macs = [], []
    for f in range(count):
        start = 100+f*100
        for bank in range(16):
            loads.append(rich('PIM_Z_LOAD_PORT', start+bank*2, start+bank*2+2,
                              bank, f, f % 2))
        for slot in range(slots):
            macs.append(native(clock=start+32+slot*16,
                byte=A.DOWN_BASE+(f*16+slot)*32, pim_buffer=f % 2))
    return loads, macs


def batch_dataset():
    events, commands = [], []
    for channel in range(16):
        for tile in range(4):
            for fragment in range(16):
                generation, buffer = tile*8+fragment//2+1, fragment % 2
                start = 100+(tile*16+fragment)*1000
                for bank in range(16):
                    extra = dict(Channel=channel, generation=generation)
                    events.append(rich('PIM_Z_LOAD', start+2*bank, start+2*bank+25,
                                       bank, fragment, buffer, **extra))
                    port = rich('PIM_Z_LOAD_PORT', start+60+2*bank, start+62+2*bank,
                                bank, fragment, buffer, **extra)
                    events.append(port)
                events.append(dict(port, operation='PIM_Z_BATCH_READY', bank_mask=65535))
                for slot in range(16):
                    byte = A.DOWN_BASE+((tile*16+fragment)*16+slot)*32
                    command = native(clock=start+100+slot*16, byte=byte,
                                     Channel=channel, pim_buffer=buffer)
                    commands.append(command)
                    event = rich('PIM_MAC_AB', command['clock'], command['clock']+16, buffer=buffer,
                                 Channel=channel, generation=generation, bank_mask=65535,
                                 mac_columns=slot+1, bytes=512,
                                 Row=command['Row'], Column=command['Column'])
                    events.append(event)
    return events, commands


class IndependentFFNDramAudit(unittest.TestCase):
    def test_fixed_layout_conserves_original_weights_and_z_work(self):
        for case, expected in A.REQUIRED_COUNTS.items():
            addresses = A.expected_addresses(case)
            self.assertEqual(Counter({op: sum(rows.values()) for op, rows in addresses.items()}), expected)
            self.assertTrue(all(key[1] == 1 for rows in addresses.values() for key in rows))
        addresses = A.expected_addresses('combined_timing_chain')
        self.assertEqual(sum(addresses['PIM_Z_STORE'].values())*32, 128*1024)
        self.assertEqual(sum(addresses['PIM_Z_LOAD'].values())*32, 512*1024)
        self.assertEqual({key[4] for key in addresses['PIM_Z_STORE']}, {176})
        self.assertEqual({key[5] for key in addresses['PIM_Z_STORE']}, set(range(16)))

    def test_local_z_must_not_be_hidden_external_dq(self):
        native_commands = [dict(command=cmd) for cmd, count in
                           [('MACAB', 32768), ('WRGB', 16384), ('LOCALWR', 4096)]
                           for _ in range(count)]
        dq, internal = A.audit_native_payload(native_commands, 'gate_up_z_store')
        self.assertEqual(dq['total'], 512*1024)
        self.assertEqual(internal['gu_gather'], 512*1024)
        self.assertEqual(internal['z_return'], 128*1024)
        with self.assertRaisesRegex(AssertionError, 'external/other'):
            A.audit_native_payload(native_commands+[dict(command='WR')], 'gate_up_z_store')

    def test_real_row_conflict_and_z_allocation_are_required(self):
        opened = native('ACTAB', 1, A.DOWN_BASE)
        zread = native('LOCALRD', 100, A.Z_BASE, BankGroup=0, Bank=0)
        with self.assertRaisesRegex(AssertionError, 'closed/wrong row'):
            A.audit_native_scope_rows([opened, zread])
        pre = native('PREpb', 50, A.DOWN_BASE, BankGroup=0, Bank=0)
        activate = native('ACT2', 60, A.Z_BASE, BankGroup=0, Bank=0)
        self.assertEqual(A.audit_native_scope_rows([opened, pre, activate, zread]), 1)
        with self.assertRaisesRegex(AssertionError, 'overwrites weights/attention'):
            A.audit_native_scope_rows([dict(zread, Row=40)])
        with self.assertRaisesRegex(AssertionError, 'rank0'):
            A.audit_native_scope_rows([dict(zread, Rank=0)])

    def test_streamed_store_allows_other_bank_write_recovery_overlap(self):
        gathers, returns, stores = streamed_stores()
        self.assertEqual(ports(gathers, returns, stores=stores, producer_ready={0: 100}), 4)
        with self.assertRaisesRegex(AssertionError, 'before complete GU'):
            ports(gathers, returns, stores=stores, producer_ready={0: 101})
        early = [dict(returns[0], clock=107, ready_at=109), returns[1]]
        with self.assertRaisesRegex(AssertionError, 'before complete G/U'):
            ports(gathers, early, stores=stores)
        early = [dict(stores[0], clock=109, ready_at=150), stores[1]]
        with self.assertRaisesRegex(AssertionError, 'before Z return'):
            ports(gathers, returns, stores=early)

    def test_load_cannot_use_issue_clock_as_data_ready(self):
        read = rich('PIM_Z_LOAD', 100, 125, buffer=0, generation=1)
        load = rich('PIM_Z_LOAD_PORT', 125, 127, buffer=0, generation=1)
        self.assertEqual(ports(loads=[load], reads=[read]), 1)
        with self.assertRaisesRegex(AssertionError, 'before LOCALRD data completion'):
            ports(loads=[dict(load, clock=124, ready_at=126)], reads=[read])
        with self.assertRaisesRegex(AssertionError, 'native Z array completion'):
            ports(loads=[load], reads=[dict(read, ready_at=100, service_cycles=0)])

    def test_rank_port_serializes_store_return_and_load_gather(self):
        gathers, returns, stores = streamed_stores()
        read = rich('PIM_Z_LOAD', 95, 120, bank=2, buffer=0, generation=1)
        load = rich('PIM_Z_LOAD_PORT', 120, 122, bank=2, buffer=0, generation=1)
        ports(gathers, returns, [load], stores, [read])
        early_read = dict(read, clock=94, ready_at=119)
        early_load = dict(load, clock=119, ready_at=121)
        with self.assertRaisesRegex(AssertionError, 'Overlapping transfers'):
            ports(gathers, returns, [early_load], stores, [early_read])

    def test_one_bank_staging_and_one_nonlinear_slice_are_bounded(self):
        gathers, returns, stores = streamed_stores()
        # Both returned vectors now target the same bank and the first write is delayed.
        gathers[1] = dict(gathers[1], Bank=0, bank_mask=1, Column=1)
        returns[1] = dict(returns[1], Bank=0, bank_mask=1, Column=1)
        stores[1] = dict(stores[1], Bank=0, bank_mask=1, Column=1)
        stores[0] = dict(stores[0], clock=120, ready_at=161)
        with self.assertRaisesRegex(AssertionError, 'bank staging buffer'):
            ports(gathers, returns, stores=stores)
        # Nonoverlapping port transfers can still require two resident128B G/U slices.
        gathers, returns, stores = streamed_stores()
        gathers[1] = dict(gathers[1], clock=108, ready_at=116)
        returns[0] = dict(returns[0], clock=116, ready_at=118)
        stores[0] = dict(stores[0], clock=118, ready_at=159)
        with self.assertRaisesRegex(AssertionError, 'nonlinear staging'):
            ports(gathers, returns, stores=stores)

    def test_all16_banks_ready_and_prior_input_buffer_dead_before_use(self):
        loads, macs = fragment_sample()
        A.audit_load_fragment_dependencies(loads, macs, fragments=3, slots=2)
        bad = copy.deepcopy(loads)
        bad[15]['ready_at'] += 1
        with self.assertRaisesRegex(AssertionError, 'before all16'):
            A.audit_load_fragment_dependencies(bad, macs, fragments=3, slots=2)
        bad = copy.deepcopy(loads)
        bad[15].update(BankGroup=bad[14]['BankGroup'], Bank=bad[14]['Bank'])
        with self.assertRaisesRegex(AssertionError, 'Missing/duplicated bank'):
            A.audit_load_fragment_dependencies(bad, macs, fragments=3, slots=2)
        bad = copy.deepcopy(loads)
        bad[32]['clock'] = macs[1]['clock']+15
        with self.assertRaisesRegex(AssertionError, 'before its previous MAC'):
            A.audit_load_fragment_dependencies(bad, macs, fragments=3, slots=2)

    def test_rich_completion_requires_matching_native_array_issue(self):
        store = rich('PIM_Z_STORE', 100, 141)
        command = native('LOCALWR', 100, A.Z_BASE, BankGroup=0, Bank=0, pim_buffer=-1)
        A.audit_rich_identity([command], [store])
        with self.assertRaisesRegex(AssertionError, 'matching actual native'):
            A.audit_rich_identity([command], [dict(store, clock=101, ready_at=142)])

    def test_batch_generations_and_masks_are_checked_against_actual_work(self):
        events, commands = batch_dataset()
        self.assertEqual(A.audit_batches(events, commands), 16*4*16)
        batch = next(e for e in events if e['operation'] == 'PIM_Z_BATCH_READY')
        batch['bank_mask'] = 32767
        with self.assertRaisesRegex(AssertionError, 'Batch readiness'):
            A.audit_batches(events, commands)
        batch['bank_mask'] = 65535
        mac = next(e for e in events if e['operation'] == 'PIM_MAC_AB')
        mac['generation'] += 1
        with self.assertRaisesRegex(AssertionError, 'lacks16'):
            A.audit_batches(events, commands)


if __name__ == '__main__':
    unittest.main()
