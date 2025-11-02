# integrate_ensemble.py
def integrate_ensemble_into_system():
    from models.ensemble.ensemble_ai import SimpleEnsembleAI
    return SimpleEnsembleAI()

if __name__ == "__main__":
    ai = integrate_ensemble_into_system()
    print("Ensemble AI integration test passed!")
