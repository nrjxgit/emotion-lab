# Dormant synthetic-only guards (S28)

No real dataset, model, launch, rights change or operator approval is enabled.
Only manufactured `synthetic-actor-N` identities and the guarded synthetic
adapter are admitted. Constructors/imports perform no application data IO.
Python import machinery still reads executable source; the no-IO test is not
OS tracing or an import sandbox. Ordinary reviewed SyntheticAdapter output remains unchanged; evaluate now rejects
subclasses and instance-injected methods. The separate GuardedSyntheticAdapter requires a capability for
records; the old evaluator cannot implicitly decode these guarded items.

## Exposure ledger

Construct ExposureLedger with an explicit absolute existing directory and a
`synthetic:<cohort>` namespace. One directory contains one namespace, shared
across ALL protocol hashes/stages; never choose a new root to evade history.
Initialization is a separate explicit `initialize(explicit_genesis=True)` call
and exclusive creation. Missing, inaccessible, empty, partial, malformed,
schema/chain-inconsistent history is refused, never interpreted as unused.
No constructor creates the directory, lock or ledger. Directory values must be
strings or string-valued PathLike objects; malformed values give a typed refusal.
Reserve accepts at most 512 input actors, checked before any write; encoded
append size is checked against the history limit before writing, preventing
an oversized request from poisoning otherwise valid history.
The 10 MiB (10,485,760 byte) history limit is a hard lifetime cap per namespace.
There is no rotation: once capacity is exhausted, further reservations refuse
with LEDGER_CAPACITY. A supervised archive procedure is future work; choosing a
new namespace/root to evade exposure history is not a recovery procedure.

The persistent lock file is protected by a Windows msvcrt byte-range lock or
POSIX flock. It is never deleted. Nonblocking acquisition has a finite timeout;
lock contention refuses rather than running unlocked. OS owner death releases
the lock: there is no PID/age-based stale-file reclaim and no deletion race.
Reservation history survives that release. Partial writes require supervised
recovery; this package never repairs/truncates/deletes history. OS primitives
are exercised only on local synthetic scratch files, not certified on all
filesystems, network shares, hostile processes or a real run identity.

The lock covers history read, schema/hash-chain validation, intersection check,
append, fsync and readback. It is released after durable reservation, not held
through decode: reservation itself excludes reuse. Every stage conservatively
consumes ALL requested actors, including pilot/DEV, not merely HELD_OUT. Thus
requested held-out actors cannot overlap any prior exposure, and a previously
held-out actor cannot migrate into another stage. Actor aliases normalize to
one synthetic numeric identity within the cohort, independent of protocol.
Failure/crash/empty output never unspends a committed reservation. A torn
reservation blocks subsequent work instead of releasing actors. There is no
decode before a successful reservation in a future integrated procedure;
this dormant package does not implement or authorize such a real procedure.

An ExposureOverride must name exactly the intersecting canonical actors, every
prior reservation involving them, an unused synthetic override ID and a bounded
reason. It is appended, not used to remove prior records, and cannot be replayed.
This synthetic exception record is NOT authenticated operator authority.

Records have contiguous sequence numbers, previous digests, closed schemas and
canonical SHA256 chaining. Parsing splits only on byte LF and compares every
record to its canonical ASCII JSON encoding: CRLF, form-feed separators and
space-padded lines refuse. Caller-supplied LedgerCheckpoint checks a known
high-water prefix; it permits legitimate subsequent appends but refuses a
missing/changed checkpoint. Trustworthy up-to-date external checkpoint custody,
one canonical directory identity, anti-fork/restore enforcement, permissions,
reparse/link/replacement defenses and hardware durability are NOT established
here. Without a current external checkpoint, truncation to an internally valid
older prefix cannot be detected. fsync/readback is an application request, not
proof of power-loss-safe storage. `require_checkpoint=True` makes a valid external
checkpoint mandatory before reading/committing; `require_external_checkpoint` is
the pure presence/shape helper. Genesis is a separate operation without that
option, followed by externally recording its checkpoint. Without the mandatory
option a rollback to genesis can be accepted; even with it, a stale genesis-only
checkpoint cannot detect later discarded reservations. A CURRENT independently
custodied high-water checkpoint is essential; the flag does not prove freshness.
External real-run controls remain HOLD.

## Capabilities and close

SyntheticPayloadSession is **self-trusting until supervisor exists**: anyone can
construct it and issue synthetic tokens without a ledger reservation, authority
or protocol hash. The explicit `SELF_TRUSTED_SYNTHETIC_ONLY` label and test record
that choice; it is not exposure enforcement. Future supervised admission must
validate the same actors/stage/protocol against a durable ledger reservation
and trusted authority before any real decoder can be wired.

It binds a caller-declared actor tuple, stage and monotonic expiry. Constructor
samples the clock and rejects expired values or TTL above 600 seconds, including
wall-clock epoch values under the default monotonic clock. Custom clocks are
synthetic test dependencies, not authenticated time. Sessions cannot be copied,
deepcopied or pickled, so ordinary copying cannot fork revocation state. Each issued object is bound by identity to one inert item and session
epoch, retained in the session's issuance registry and not pickle/JSON based.
Each records/decode call checks live session, stage, actor, item and registered
capability plus the issuance-time metadata snapshot before generating manufactured
records, then checks live state and snapshot again before returning. Tokens are
single-use, consumed immediately before generation (including failed generation).
Items are immutable NamedTuples: object.__setattr__ cannot mutate their fields;
issuance additionally retains a metadata snapshot. Expiry, clock rollback or revoke/close prevents further use.
No arbitrary decoder callback or real adapter is wired. Arbitrary in-process
Python can still subvert private objects; these are correctness boundaries,
not a sandbox. Direct __reduce__()/__getstate__() manipulation can still expose
session private state; manually rebuilding from that state can fork revocation.
That is the same arbitrary in-process/private-state manipulation limit, not a
supported copy API. No streaming native decoder is implemented or claimed.

verify_grant_closed is a pure check on a complete parsed allowlist snapshot:
all known real declarations must be present and not granted; extra entries
also cannot be granted. Only the synthetic declaration may remain granted.
Blocked-pending entries must additionally have explicitly empty allowed_uses
and scopes; malformed/unhashable status gives a typed refusal. The shipped
allowlist retains blocked entries with nonempty declarations, so it does NOT
pass this stricter close predicate; no allowlist/pin is changed here.
It performs no rollback, cache refresh, process drain, epoch broadcast or
authority verification. Future procedures must revoke sessions, drain ALL
grant-bearing processes and obtain an independently verified fresh-process
snapshot under the restored pin before further data work. Failed/abandoned
close remains HOLD. No grant/allowlist/config pin is changed by this package.

The legacy evaluate uses one authorization Decision for metadata admission and
provenance, without a second authorize in items() or manifest creation. Honest
non-synthetic adapters are refused. Only `type(adapter) is SyntheticAdapter`
with exact synthetic name and reviewed constructor fields is accepted; no
subclass/instance-injected data method enters evaluation. Unreviewed in-process
code impersonating synthetic (including monkeypatching reviewed code/classes)
is outside this boundary. This is not an in-process sandbox or authenticated
code provenance. Base `_items`/_participants cannot be made final without
breaking concrete adapter implementations, so rights.py is left unchanged.
