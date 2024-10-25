# Spectral similarity evaluation

Experimental framework to evaluate spectral similarities

## Installation

After having cloned the repository, install the required packages with the following command:

```bash
pip install -r requirements.txt
```

## Usage

From the root directory of the repository, run the following command to test that everything is working and executing a smoke test.

```bash
pytest -sx
```

Next, once you have ensured that the tests are passing, you can run the main script with the following command:

```bash
python run.py --iterations 10\
    --quantity 10000\
    --random-state 67455636\
    --data-directory "data"\
    --output "results.csv"\
    --verbose
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
