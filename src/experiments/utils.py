import datetime
import os

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                           os.pardir)

EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, 'experiments')
SAMPLES_PATH = os.path.join(PROJECT_DIR, 'src', 'furnishing', 'room_samples.pkl')

if not os.path.exists(EXPERIMENTS_DIR):
    os.makedirs(EXPERIMENTS_DIR)


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_experiment_dir(comparison_dir=False):
    timestamp = datetime.datetime.now().strftime('%HH_%MM_%dd_%mm_%yy')
    if comparison_dir:
        timestamp += '_comparison'
    joined = os.path.join(EXPERIMENTS_DIR, timestamp)
    ensure_dir(joined)
    csv_dir = os.path.join(joined, 'csvs')
    latex_dir = os.path.join(joined, 'latexs')
    plots_dir = os.path.join(joined, 'plots')
    ensure_dir(csv_dir)
    ensure_dir(latex_dir)
    ensure_dir(plots_dir)
    return joined, csv_dir, latex_dir, plots_dir
