# Durability and Bitcoin Publication Policy

Status: Profile 1 publication policy

Review date: 2026-07-25

## Purpose

This document defines the durability claim for Wavey Profile 1.

It defines the Bitcoin record, decoder capsule, verification evidence, and recovery limits.

The [lineage contract](encodec-lineage-and-lm-contract.md) defines the codec and entropy requirements.

Profile 1 is a clean break. It has no backward compatibility with unpublished Wavey formats.

## Requirement words

In this document, **MUST** identifies a Profile 1 requirement.

**SHOULD** identifies a recommended property.

**MAY** identifies a permitted option.

## Terms

A **media object** is one complete Profile 1 encoded audio object.

A **profile manifest** defines all decode rules and identifies all decode artifacts.

A **profile root** is the SHA-256 digest of the canonical profile manifest.

An **object manifest** lists the media object, decoder capsule, and their exact byte lengths and digests.

An **object root** is the SHA-256 digest of the canonical object manifest.

A **decoder capsule** is a closed package that contains all decode-critical artifacts.

A **capsule root** is the SHA-256 digest of the canonical capsule manifest.

A **Bitcoin commitment** is data in a Bitcoin transaction that commits to declared digests.

A **Bitcoin-contained object** has all declared object bytes in one or more Bitcoin transactions.

A **recovery test** builds and runs a decoder in an isolated environment.

A **canonical decode** is a strict decode that passes all Profile 1 integrity checks.

A **salvage decode** replaces or omits damaged ranges and reports those changes.

A **software bill of materials** lists the software components in a release.

## Credible claim

Wavey MAY make this claim after all release gates pass:

> Wavey Profile 1 provides a published and independently tested recovery package. The cited Bitcoin record commits to its exact media, manifest, and decoder-capsule roots.

Wavey MUST add this condition near the claim:

> Recovery depends on access to the required object bytes, capsule bytes, and a usable conforming execution environment.

For a Bitcoin-contained object, Wavey MAY make this additional claim:

> The cited Bitcoin transactions contain the declared Profile 1 bytes. Two independent decoders recovered the declared codes on the stated test date.

Wavey MUST state the test date and evidence identifier.

Wavey MUST NOT promise perpetual or guaranteed decoding.

Wavey MUST NOT describe Profile 1 audio as lossless.

Wavey MUST NOT describe a Profile 1 object as a Meta ECDC bitstream.

## Claim classes

Wavey MUST use the following claim classes without substitution.

| Claim class | Minimum condition | Permitted meaning |
| --- | --- | --- |
| `bitcoin-committed` | A confirmed transaction commits to the object root. | Bitcoin verifies the declared digest relationship. |
| `bitcoin-contained` | Bitcoin transactions contain every declared object byte. | A conforming extractor can recover the object bytes from block data. |
| `recovery-tested` | Two independent decoders pass the release test. | The named capsule worked in the recorded environments on the test date. |
| `durable-package` | All three earlier conditions pass. | The object has a committed, contained, and tested recovery package. |

A claim class describes evidence at a specified time.

It does not predict the state of future software, hardware, law, or data access.

## Profile 1 clean break

Profile 1 MUST use a new format identifier and profile root.

Profile 1 decoders MUST reject `acv=2`, `acv=3`, and `acv=4`.

Profile 1 decoders MUST not guess a format from payload contents.

Compatibility tools MAY read an earlier format outside the Profile 1 decoder.

Profile 1 evidence MUST not identify an earlier Wavey payload as conforming.

Wavey SHOULD create Profile 1 objects from the lossless master.

Wavey SHOULD not create them by transcoding an earlier lossy Wavey object.

## What Bitcoin commits

A confirmed Bitcoin transaction records commitment data in an accepted block history.

A verifier can compare available bytes with the declared digests.

A successful comparison identifies the committed bytes.

The block height and block hash identify the relevant chain position.

The transaction identifier and carrier location identify the commitment data.

The block time is approximate. It MUST NOT serve as a precise creation time.

Bitcoin confirmations reduce reorganization risk. They do not give absolute finality.

The publication policy MUST declare its required confirmation count.

Wavey MUST monitor the record until it reaches that count.

Wavey MUST issue replacement evidence after a reorganization removes the record.

### Digest-only commitment

A digest-only commitment contains the object root but not the complete object.

It proves object identity when a verifier already has the object manifest.

It does not preserve the media object, capsule, weights, source, or test vectors.

A URL and digest do not make an object Bitcoin-contained.

Wavey MUST use the `bitcoin-committed` claim for this publication mode.

### Bitcoin-contained publication

A Bitcoin-contained publication MUST include every declared byte.

It MUST include the canonical object manifest.

Its object manifest MUST define transaction order, byte ranges, and complete object length.

Each stored part MUST have a byte length and SHA-256 digest.

The manifest MUST define extraction without one named wallet or indexer.

The capsule MUST include a small extraction tool and a complete extraction specification.

Wavey MUST test extraction from two independent block-data sources.

Many media objects MAY reference one Bitcoin-contained capsule root.

Each object manifest MUST identify that capsule root and its Bitcoin locations.

Pruned nodes can omit old block data.

Wallets, remote procedure call interfaces, and indexers can also hide carrier data.

Therefore, Bitcoin-contained publication does not guarantee convenient future access.

## What Bitcoin does not preserve

Bitcoin consensus does not validate codec semantics.

Bitcoin consensus does not run the decoder.

Bitcoin consensus does not verify audio quality.

Bitcoin consensus does not prove that Wavey created the committed object.

Bitcoin consensus does not make an off-chain capsule available.

Bitcoin consensus does not preserve an undisclosed decryption key.

Bitcoin consensus does not grant model, music, or software rights.

A Bitcoin commitment cannot repair omitted or incorrect release artifacts.

A publisher signature supplies origin evidence. It does not prove technical correctness.

## Canonical manifests

Profile 1 MUST define one canonical serialization for each manifest.

Each manifest entry MUST contain a path, role, byte length, and SHA-256 digest.

The object manifest MUST contain the profile root, media digest, capsule root, and Bitcoin publication mode.

The object manifest MUST identify the encoded sample count and channel order.

The object manifest MUST identify the exact Profile 1 format identifier.

The capsule manifest MUST identify every decode-critical file.

The profile manifest MUST identify the exact neural and LM weight digests.

It MUST identify the pinned Meta source commit and the Wavey source changes.

It MUST define all Wavey preprocessing and postprocessing rules.

A model name or upstream commit does not identify a complete decoder.

The manifests MUST not depend on map order, local paths, locale, or file timestamps.

Wavey MAY add a newer digest algorithm in a later evidence record.

The later record MUST retain the original SHA-256 values.

## Decoder capsule

The decoder capsule MUST form a closed decode set.

Decode-critical artifacts MUST not require Wavey DNS, accounts, APIs, license servers, secrets, or network access.

A URL and hash do not satisfy capsule closure.

Wavey MUST resolve distribution rights before publication.

The capsule MUST contain these artifacts:

| Artifact | Required content |
| --- | --- |
| Capsule manifest | Canonical file list, roles, lengths, and SHA-256 digests |
| Profile specification | Complete container, entropy, neural, and reconstruction rules |
| Reference decoder | Buildable source for the canonical decoder |
| Independent decoder | Buildable source for a separately implemented decoder |
| Model artifacts | Exact neural decoder graph, weights, tensor layouts, and operation rules |
| Entropy artifacts | Exact LM weights, entropy tables or rules, and arithmetic rules |
| Build set | Locked dependencies, build instructions, and toolchain definitions |
| Portable executables | Tested binaries or WebAssembly for supported targets |
| Extraction set | Bitcoin extraction specification and source |
| Test vectors | Valid, boundary, malformed, and corruption vectors |
| Expected results | Exact recovered-code digests and canonical PCM results |
| Lineage record | Upstream commit, Wavey changes, model origin, and exact weight digests |
| Legal files | Licenses, notices, model origin, and source origin |
| Evidence set | Signatures, public keys, release report, and prior verification reports |
| Operator guide | Offline extraction, build, verify, decode, and salvage procedures |

The source package MUST include all generated-file inputs.

The capsule MUST not require a download from the original Meta repository.

The build set MUST include checksums for all tools and dependencies.

Portable executables do not replace buildable source.

The decoder capsule MAY omit the encoder.

The capsule MUST label the encoder as optional when it includes one.

The capsule MUST include a software bill of materials.

The capsule SHOULD include source archives for permitted third-party dependencies.

## Independent implementations

Profile 1 MUST have two independent decoders before public release.

The decoders MUST use separate source trees.

They MUST not share container, entropy, RVQ mapping, or reconstruction code.

One decoder MUST not be generated from the other decoder.

The decoders MAY share the specification, test vectors, and immutable model weights.

The decoders SHOULD use different implementation languages or primary runtimes.

At least one decoder MUST support a documented CPU-only path.

The two decoders MUST recover identical RVQ codes from each valid vector.

They MUST also recover identical scale values as binary32 values.

One documented CPU path MUST define canonical PCM output.

Other neural backends MAY use published numerical and perceptual tolerances.

A backend difference MUST NOT change recovered RVQ codes.

A backend difference MUST NOT change strict integrity results.

## Clean recovery verification

Wavey MUST run a clean recovery test before each Profile 1 publication.

The test environment MUST start without installed Wavey software.

The test MAY download the declared Bitcoin block data and capsule before isolation.

The test MUST disable network access before build and decode.

The test MUST perform these actions:

1. Extract the object from the declared Bitcoin locations.
2. Verify transaction locations and all manifest digests.
3. Verify release and evidence signatures.
4. Build both decoders only from capsule contents.
5. Run every conformance and corruption vector.
6. Decode the publication media object with both decoders.
7. Compare recovered RVQ codes and scale values.
8. Compare PCM with the declared canonical result or tolerance.
9. Produce and sign a recovery evidence report.

Wavey MUST repeat the test quarterly during the first publication year.

Wavey MUST repeat the test at least once in each later calendar year.

Wavey MUST also repeat it after a decoder, toolchain, carrier, or storage change.

Wavey MUST repeat it after a reported recovery failure.

An independent maintainer SHOULD run one verification each year.

Wavey MUST publish each report without replacing an earlier report.

A failed recurring test suspends the `recovery-tested` and `durable-package` claims.

Wavey MAY restore those claims after a new complete test passes.

## Lossless-master policy

EnCodec is a lossy codec.

Wavey MUST keep a lossless production master when rights permit preservation.

The master record MUST include format, sample rate, bit depth, channel map, length, and SHA-256 digest.

Wavey SHOULD use Broadcast Wave, WAV, or FLAC with documented settings.

Wavey MUST not treat a Profile 1 object as the lossless master.

Wavey MUST keep at least three verified master copies.

The copies SHOULD use two independent storage systems.

At least one copy SHOULD use offline or immutable storage.

Wavey MUST run a master fixity check at least once each year.

A Bitcoin digest of the master proves identity. It does not preserve off-chain master bytes.

Wavey MAY publish the complete lossless master on Bitcoin when rights and privacy permit it.

If no lossless master remains, Wavey MUST record that condition.

In that case, Wavey MUST describe Profile 1 as a lossy preservation copy.

## Resilience and salvage

Independent chunks limit the effect of some local damage.

CRC detects accidental changes. It does not correct them.

SHA-256 identifies the complete expected object. It does not reconstruct missing bytes.

Profile 1 MUST define strict and salvage behavior separately.

Strict decode MUST stop and report invalid, missing, duplicate, reordered, or truncated chunks.

Salvage decode MAY replace only declared damaged ranges.

Salvage output MUST identify each replaced source range.

Wavey MUST not publish salvage output as canonical PCM.

The object manifest MAY identify optional parity or replication objects.

Any parity method MUST have a versioned specification and test vectors.

## Recovery limits

Recovery can fail when required bytes are absent.

Recovery can fail when an extraction method is incomplete.

Recovery can fail when no available environment can build or run either decoder.

Recovery can fail when legal controls prevent access to required artifacts.

Encrypted objects can become unrecoverable after key loss.

A public durability claim MUST NOT depend on an undisclosed key.

Chunk checksums cannot repair damage without parity or another valid copy.

Neural output can differ slightly across conforming floating-point backends.

The canonical CPU path controls exact PCM evidence.

Alternative backends control only their published tolerance results.

Profile 1 reconstruction does not reproduce the lossless master.

The decoder capsule reduces these risks. It cannot remove all future risk.

## Signed evidence

Wavey MUST sign the profile manifest, object manifest, capsule manifest, and each recovery report.

Each signature MUST cover a domain-separated statement.

The statement MUST include the manifest type, schema version, digest, and byte length.

The signature record MUST identify its algorithm, public key, and key identifier.

The capsule MUST contain signature verification source and test vectors.

The object manifest MAY include artist, label, archive, or independent-maintainer signatures.

Each recovery report MUST contain:

| Field | Required value |
| --- | --- |
| Evidence identity | Schema version, report digest, and prior report digest |
| Test time | UTC time and monotonic test duration |
| Profile identity | Format identifier and profile root |
| Object identity | Object root, media digest, byte length, and sample count |
| Capsule identity | Capsule root and byte length |
| Bitcoin location | Network, block height, block hash, transaction identifier, and carrier ranges |
| Chain state | Confirmation count at report creation |
| Decoder identity | Source digest, build digest, version, and implementation name |
| Environment | Architecture, operating system, compiler, runtime, and CPU features |
| Test coverage | Vector identifiers and corruption cases |
| Exact results | Recovered-code, scale, and canonical-PCM digests |
| Tolerance results | Maximum error and declared perceptual metrics for other backends |
| Signer identity | Signature algorithm, key identifier, public key, and signature |

The Bitcoin transaction proves inclusion of commitment data.

The Wavey signature proves control of the declared release key.

The recovery report records a successful test at one time.

None of these facts proves future executability.

Wavey MUST publish signed key-rotation and key-revocation records.

A later report MUST link to the prior report digest.

Wavey SHOULD commit each annual evidence-chain head to Bitcoin.

## Publication procedure

1. Freeze the Profile 1 specification and create its profile root.
2. Create the media object from the verified lossless master.
3. Build the closed decoder capsule.
4. Create canonical capsule and object manifests.
5. Run both decoders in isolated environments.
6. Sign the manifests and release evidence.
7. Publish the declared commitment or contained bytes to Bitcoin.
8. Extract the publication from two independent block-data sources.
9. Wait for the declared confirmation count.
10. Sign and publish the final Bitcoin evidence record.
11. Copy the capsule and evidence to independent content-addressed archives.
12. Schedule the next clean recovery test.

Wavey MUST stop publication when a required step fails.

Wavey MUST preserve failed evidence for diagnosis.

Wavey MUST issue a new profile root after any decode-semantic change.

Wavey MAY issue a new object root after a packaging-only change.

The new manifest MUST identify the unchanged profile root in that case.

## Release gate

Wavey can use the `durable-package` claim only when all answers are yes:

- Does Bitcoin contain the declared media and capsule bytes?
- Can two independent sources extract identical bytes?
- Do all manifest digests and signatures verify?
- Can two independent decoders build without network access?
- Do both decoders recover identical RVQ codes and scales?
- Does the canonical CPU decoder match the declared PCM digest?
- Do other supported backends meet their declared tolerances?
- Do strict corruption tests reject every invalid object?
- Does salvage report every replaced range?
- Does a verified lossless master remain available?
- Does the signed evidence identify the complete test environment?
- Has the Bitcoin record reached the declared confirmation count?

A “no” answer limits the claim to the highest completed claim class.
