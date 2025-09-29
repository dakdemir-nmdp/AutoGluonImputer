# Contributing to AutoFillGluon

Thank you for considering contributing to AutoFillGluon! This document outlines the process for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct. Please be respectful and considerate of others.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with the following information:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any relevant logs or screenshots
- Your environment (Python version, operating system, etc.)

### Suggesting Enhancements

We welcome suggestions for enhancements. Please create an issue on GitHub with the following information:

- A clear, descriptive title
- A detailed description of the proposed enhancement
- Any relevant examples or use cases
- If applicable, potential implementation details or strategies

### Contributing Code

1. Fork the repository
2. Create a new branch for your feature or bugfix (`git checkout -b feature/your-feature` or `git checkout -b fix/your-bugfix`)
3. Make your changes
4. Add or update tests as necessary
5. Run the tests to ensure they pass
6. Update documentation as necessary
7. Commit your changes (`git commit -m 'Add some feature'`)
8. Push to the branch (`git push origin feature/your-feature`)
9. Open a Pull Request

#### Pull Request Guidelines

- Include a clear, descriptive title
- Reference any relevant issues
- Include a summary of the changes and their motivation
- If you've added code that should be tested, add tests
- Ensure the test suite passes
- Make sure your code follows the existing style (using a linter can help)
- Update documentation as necessary

## Development Setup

1. Clone the repository
2. Install the package in development mode:

```bash
pip install -e .
```

Or use the provided Makefile:

```bash
make install
```

3. Run the tests to ensure everything is working:

```bash
python -m unittest discover tests
```

Or:

```bash
make test
```

## Style Guide

We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.

## Testing

Please ensure all tests pass before submitting a Pull Request. If you're adding new features, please add appropriate tests.

To run tests:

```bash
make test
```

## Documentation

Please update documentation as necessary. This includes:

- Docstrings for new functions, classes, and methods
- Updates to example code
- Updates to README.md or other documentation files

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).