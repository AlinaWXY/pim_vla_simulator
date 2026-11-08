// Issue-to-issue constraints from LPDDR5XPIM(pim_data_path="attacc"),
// LPDDR5X_8533. These are independent of delayed DQ return reservations.
// The native-spec differential test checks every transfer pair and scope.
static I transfer_issue_gap(const OnlineJob& a,const OnlineJob& b) {
  if(!a.dq_length || !b.dq_length)return 0;
  const bool same_rank=a.rank==b.rank;
  const bool same_bg=same_rank && (a.bg<0 || b.bg<0 || a.bg==b.bg);
  // AttAcc adds Channel nBL_min, Rank nCCDS, and BankGroup nCCDL
  // for ALL WRGB/RDRES/RD/RDA/WR/WRA combinations, including cross-rank.
  I gap=(same_bg?4:2)*CK;
  // Ordinary array RD/WR also retain the inherited LPDDR5X constraints.
  // Buffer transfers have zero latency in this native AttAcc configuration
  // and must not inherit CL/CWL-based ordinary-command issue formulas.
  if(a.array_access && b.array_access) {
    I normal;
    if(!same_rank) {
      normal=a.write?(b.write?3:1):(b.write?15:4);
    } else if(a.write && !b.write)normal=same_bg?29:21;
    else if(!a.write && b.write)normal=same_bg?17:15;
    else normal=same_bg?4:2;
    gap=std::max(gap,normal*CK);
  }
  return gap;
}

// DQ end-to-start guards in uniform_transactions.cpp are a separate
// conservative bus model. For ordinary transfers they are equivalent to
// native LPDDR5X turnarounds. For mixed buffer/array transfers the native
// AttAcc spec has no corresponding cross-rank guard: retain the existing
// 1/2 CK extension explicitly, rather than call it a native timing rule.
struct CommittedTransfer { OnlineJob job; I issue; bool gpu; };
