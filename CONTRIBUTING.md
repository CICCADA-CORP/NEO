# Contributing to NEO

Thank you for your interest in contributing to NEO! This project aims to create the future of audio formats, and we need help from developers, audio engineers, ML researchers, and specification writers.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Specification Changes](#specification-changes)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. Be respectful, inclusive, and constructive.

## How to Contribute

### 🐛 Bug Reports
- Use GitHub Issues with the `bug` label
- Include: steps to reproduce, expected vs actual behavior, OS/Rust version
- Attach sample `.neo` files if relevant

### 💡 Feature Requests
- Use GitHub Issues with the `enhancement` label
- Reference the relevant section of [SPECIFICATION.md](./SPECIFICATION.md)
- Describe the use case and proposed solution

### 🔧 Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Write code + tests
4. Run checks: `cargo fmt && cargo clippy && cargo test`
5. Submit a Pull Request

### 📝 Specification Changes
- Open a GitHub Issue with the `spec` label first
- Discuss the change with maintainers before writing
- Spec changes require at least 2 approvals

## Development Setup

### Prerequisites
- **Rust 1.75+**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Python 3.10+**: For neural codec components
- **Git**: For version control

### Building

```bash
# Build all crates
cd crates
cargo build

# Run all tests
cargo test

# Run with all checks
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

### Python Setup (Neural Components)

```bash
cd neural
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```
crates/
├── neo-format/     # Container format (START HERE for container work)
├── neo-codec/      # Audio codecs (START HERE for codec work)
├── neo-metadata/   # Metadata handling
├── neo-spatial/    # Spatial audio
├── neo-stream/     # Streaming/P2P
├── neo-edit/       # Non-destructive editing
├── neo-cli/        # CLI tool
└── neo-ffi/        # C FFI bindings
```

## Coding Standards

### Rust
- **Edition**: 2021
- **Formatting**: `rustfmt` with default settings
- **Linting**: `clippy` with `-D warnings`
- **Error handling**: Use `thiserror` for library errors, `anyhow` for binary/CLI
- **Documentation**: All public items MUST have doc comments
- **Testing**: All new functionality MUST have tests
- **Naming**: 
  - Types: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `SCREAMING_SNAKE_CASE`
  - Crates: `neo-*` (kebab-case)

### Python
- **Version**: 3.10+
- **Formatting**: `ruff format`
- **Linting**: `ruff check`
- **Type hints**: Required for all public functions
- **Docstrings**: Google style

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add FLAC residual encoding to neo-codec
fix: correct BLAKE3 hash computation for empty chunks
docs: update SPECIFICATION.md section 7.1
test: add round-trip tests for 96kHz audio
chore: update ort dependency to 2.1
refactor: extract chunk validation into separate module
```

### Scope prefixes:
- `format:` — neo-format crate
- `codec:` — neo-codec crate
- `meta:` — neo-metadata crate
- `spatial:` — neo-spatial crate
- `stream:` — neo-stream crate
- `edit:` — neo-edit crate
- `cli:` — neo-cli crate
- `spec:` — specification changes
- `neural:` — Python ML components

## Testing

### Running Tests
```bash
# All tests
cd crates && cargo test

# Specific crate
cargo test -p neo-format

# With output
cargo test -- --nocapture

# Just the integration tests
cargo test --test '*'
```

### Test Guidelines
- Unit tests: co-locate in the source file with `#[cfg(test)]` modules
- Integration tests: place in `tests/` directory at workspace root
- Test fixtures: place audio files in `tests/fixtures/`
- Use `proptest` or `quickcheck` for property-based testing where appropriate
- Use `cargo-fuzz` for fuzzing parsers (especially neo-format reader)

### Audio Quality Testing
- Use PESQ (Perceptual Evaluation of Speech Quality) for speech
- Use ViSQOL for music
- Compare against reference codecs (Opus, AAC) at equivalent bitrates
- Document results in PR descriptions

## Specification Changes

The specification ([SPECIFICATION.md](./SPECIFICATION.md)) is the source of truth. Changes require:

1. **Issue first**: Open a GitHub Issue with `spec` label describing the proposed change
2. **Discussion period**: At least 7 days for community feedback
3. **PR with rationale**: Include a clear explanation of why the change is needed
4. **Two approvals**: At least 2 maintainers must approve
5. **Version bump**: Increment the spec draft version
6. **Implementation plan**: Describe how existing code must change

### Breaking Changes
Changes to the binary format (header, chunk table, chunk types) are considered breaking and require:
- A format version increment
- Backward-compatible reading (new readers MUST read old files)
- A migration guide

## Pull Request Process

1. **Create the PR** with a clear title and description
2. **Link issues** that this PR addresses
3. **Ensure CI passes**: `cargo fmt`, `cargo clippy`, `cargo test`
4. **Include tests** for new functionality
5. **Update documentation** if behavior changes
6. **Request review** from at least 1 maintainer
7. **Address feedback** promptly
8. **Squash merge** when approved

### PR Template
```markdown
## Description
Brief description of changes.

## Related Issues
Closes #XX

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation
- [ ] Specification change

## Checklist
- [ ] `cargo fmt` passes
- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] New code has doc comments
- [ ] New functionality has tests
- [ ] SPECIFICATION.md updated (if applicable)
```

---

Thank you for helping build the future of audio! 🎛️
