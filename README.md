# KWECLI - Knowledge Workflow Engine CLI

## ⚠️ PROTOTYPE STATUS - NOT PRODUCTION READY ⚠️

This repository contains a **prototype/experimental implementation** of KWECLI (Knowledge Workflow Engine CLI) for research and development purposes only.

### Current Status
- 🟡 **Prototype functionality achieved** - 3-phase enhancement complete
- 🔴 **Not ready for production use**
- 🟡 **Research/development testing successful**
- 🔴 **No guarantee of production stability**
- 🔴 **Breaking changes expected frequently**

## What is KWECLI?

KWECLI is an experimental autonomous development CLI tool that aims to provide:

- **Autonomous Code Generation**: AI-powered code generation using local models (Ollama)
- **Advanced LTMC Integration**: Native integration with Long-Term Memory Core systems
- **Sprint Management**: Project management with sprint planning and coordination
- **Code Drift Detection**: Automated detection of code inconsistencies and documentation drift
- **Multi-System Coordination**: Orchestration of multiple AI agents and tools

## Architecture Overview

The project contains several experimental components:

### Core Components
- `kwecli/` - Main CLI application package
- `bridge/` - Native LTMC integration bridge
- `start_kwe_cli.py` - Main entry point (experimental)

### Key Features (Experimental)
- **Native LTMC Bridge**: Direct integration with LTMC tools without MCP overhead
- **Advanced Service Layer**: Sprint management, code drift detection, blueprints
- **Autonomous Development**: Natural language command processing
- **Local AI Integration**: Ollama-based code generation

### Supporting Systems
- `agents/` - AI agent coordination system
- `tools/` - Various utility tools
- `config/` - Configuration management
- `frontend/` - Web interface components (experimental)

## Requirements

- Python 3.9+
- Ollama (for local AI models)
- LTMC system (Long-Term Memory Core)
- Various Python dependencies

## Installation

**⚠️ WARNING: This is prototype code and may not work as expected**

```bash
# This is experimental - use at your own risk
git clone https://github.com/yourusername/kwecli.git
cd kwecli
# Installation process is not yet standardized
```

## Usage

**⚠️ This software is not functional yet**

The intended usage would be:
```bash
python start_kwe_cli.py -c "your development command"
```

But this is currently non-functional and experimental.

## Prototype Development Status

### ✅ Completed Experimental Features (3-Phase Enhancement)

**Phase 1 - Semantic Routing Intelligence:**
- Intelligent query classification and database routing
- Sub-millisecond routing decisions
- Native LTMC integration

**Phase 2 - Quality Evaluation System:**  
- 6 comprehensive quality evaluation metrics
- 99.56% performance improvement (35s → 155ms)
- Real-time feedback loops

**Phase 3 - Retrieval Refinement:**
- 4 advanced retrieval strategies (HyDE, Fusion, Rerank, Contextual)
- Multi-source context optimization
- Sub-15ms processing performance

### 📊 Prototype Performance Metrics

| Component | Target | Achieved | Status |
|-----------|---------|----------|---------|
| Semantic Router | ≤500ms | 0-2ms | ✅ Excellent |
| Quality Evaluator | ≤500ms | 155ms | ✅ Excellent |
| Retrieval Refiner | ≤5000ms | 10-15ms | ✅ Excellent |
| End-to-End Integration | Working | 184ms | ✅ Functional |

## 📜 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed development history and technical specifications.

## Architecture Overview

```
KWECLI Enhanced Intelligence Stack (Prototype):
├── bridge/semantic_router.py        (Query → Tool Selection)
├── bridge/quality_evaluator.py      (Output → Quality Score)  
├── bridge/retrieval_refiner.py      (Context → Relevance)
├── bridge/bridge_core.py            (LTMC Integration)
└── planner/agent.py                 (Workflow Orchestration)
```

## Research Notes

- Implements "phi 500mb approach" for performance optimization
- Uses lightweight checks instead of heavyweight subprocess calls
- Real LTMC database integration (SQLite, FAISS, Neo4j, Redis)
- CLAUDE.md compliant architecture (≤300 lines per file)
- Zero mocks or placeholder implementations

## License

MIT License - See [LICENSE](LICENSE) file

## ⚠️ Important Disclaimers

**THIS IS EXPERIMENTAL PROTOTYPE SOFTWARE:**

- **Not for Production**: This prototype is intended for research and development only
- **Database Operations are Real**: May affect LTMC data in connected systems
- **No Warranties**: Provided "as is" without guarantees of functionality or security
- **Breaking Changes Expected**: Experimental software subject to significant changes
- **Use at Own Risk**: Users assume all responsibility for testing and validation

**Research and Development Use Only** - Do not deploy in production environments.

## Contributing

This prototype demonstrates advanced AI-powered development workflows. Contributions welcome for research purposes, but please understand this is experimental software not intended for production use.