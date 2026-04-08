# Contributing to CycleGAN and pix2pix PyTorch Implementation

We welcome contributions! Please follow these guidelines:

## Code Style

- Use [Black](https://black.readthedocs.io/) for code formatting with line length of 120
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Add docstrings to functions and classes

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Making Changes

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit with clear messages:
```bash
git add .
git commit -m "description of changes"
```

3. Push to your fork and submit a pull request

## Reporting Issues

- Use the GitHub issue tracker
- Provide a clear description of the problem
- Include code to reproduce the issue if possible
- Mention your environment (OS, Python version, PyTorch version)

## License

By contributing, you agree that your contributions will be licensed under the BSD License.
