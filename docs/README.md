# Lunar Horizon Optimizer Documentation

## Overview

This directory contains comprehensive documentation for the Lunar Horizon Optimizer project, including architecture details, development plans, testing strategies, and refactoring guidelines.

## Documentation Structure

### 📋 **Planning & Strategy**
- **[Refactoring Plan](refactoring_plan.md)** - Comprehensive plan for code restructuring and architecture improvements
- **[Development Roadmap](../scripts/PRD.txt)** - Product requirements and development phases
- **[Task Management](../tasks/)** - Current development tasks and priorities

### 🏗️ **Architecture & Design**
- **[Trajectory Modules](trajectory_modules.md)** - Detailed documentation of trajectory calculation modules
- **[Configuration System](../src/config/)** - Mission configuration and parameter management
- **[Project Structure](../README.md)** - Overall project organization and setup

### 🧪 **Testing & Quality**
- **[Test Documentation](trajectory_tests.md)** - Comprehensive testing strategy and test suite overview
- **[Test Coverage](../tests/)** - Unit and integration test implementations

### 📐 **Development Guidelines**
- **[CLAUDE.md](../CLAUDE.md)** - Project working rules and development standards
- **[Contribution Guidelines](../README.md#contributing)** - How to contribute to the project

## Quick Navigation

### For Developers
1. **New to the project?** Start with [Project README](../README.md)
2. **Understanding the codebase?** Read [Trajectory Modules](trajectory_modules.md)
3. **Making changes?** Follow [CLAUDE.md](../CLAUDE.md) guidelines
4. **Planning improvements?** Review [Refactoring Plan](refactoring_plan.md)

### For Contributors
1. **Setting up?** Follow [Installation Guide](../README.md#installation)
2. **Running tests?** See [Test Documentation](trajectory_tests.md)
3. **Understanding tasks?** Check [Task Management](../tasks/)
4. **Code quality?** Follow [Development Standards](../CLAUDE.md)

## Documentation Standards

### File Organization
- **README files** provide overviews and navigation
- **Technical documentation** includes implementation details
- **Planning documents** cover strategy and roadmaps
- **Test documentation** explains testing approaches

### Documentation Conventions
- Use clear, descriptive headers
- Include code examples where helpful
- Provide navigation links between related documents
- Keep content up-to-date with implementation changes

### Updating Documentation
- Update documentation when making code changes
- Review related docs when modifying functionality
- Ensure examples remain current and functional
- Maintain consistency across all documentation

## Key Concepts

### Trajectory Optimization
The project focuses on optimizing lunar mission trajectories by combining:
- **High-fidelity orbital mechanics** using PyKEP and PyGMO
- **Economic analysis** with ROI and NPV calculations
- **Multi-objective optimization** balancing physics and economics
- **ISRU benefit modeling** for resource extraction scenarios

### Architecture Principles
- **Modular design** with clear separation of concerns
- **Configuration-driven** mission parameters
- **Comprehensive testing** with high coverage
- **Extensible architecture** for future enhancements

### Development Workflow
1. **Plan** - Review tasks and documentation
2. **Implement** - Follow coding standards and patterns
3. **Test** - Ensure comprehensive test coverage
4. **Document** - Update relevant documentation
5. **Review** - Code review and quality checks

## Getting Help

### Technical Issues
- Check [Test Documentation](trajectory_tests.md) for testing guidance
- Review [Trajectory Modules](trajectory_modules.md) for implementation details
- Consult [CLAUDE.md](../CLAUDE.md) for development standards

### Project Questions
- Review [Development Roadmap](../scripts/PRD.txt) for project goals
- Check [Task Management](../tasks/) for current priorities
- See [Refactoring Plan](refactoring_plan.md) for architecture decisions

### Contributing
- Follow [Development Guidelines](../CLAUDE.md)
- Review [Project Structure](../README.md)
- Ensure tests pass and documentation is updated

---

*This documentation is maintained as part of the Lunar Horizon Optimizer project and should be kept current with code changes and project evolution.*