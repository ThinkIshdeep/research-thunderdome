import time
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from .core import Base, bounded, RUNTIME

class Engine(Base):
    actions = Base.actions | {'campaign', 'audit'}

    def __init__(self):
        super().__init__()
        self.reset({})

    def reset(self, p):
        self.state = {'dataset': 'breast_cancer', 'trials': [], 'audits': [], 'last_result': None}

    def audit(self, p):
        issue = p.get('issue', 'fit_before_split')
        explanations = {'fit_before_split': 'Rejected: preprocessing must be fit on training partitions only.', 'test_tuning': 'Rejected: final test feedback may not select a model.', 'arbitrary_code': 'Rejected: runner accepts allowlisted estimator configurations, not executable code.'}
        if issue not in explanations:
            raise ValueError('Unknown audit fixture')
        out = {'issue': issue, 'accepted': False, 'reason': explanations[issue]}
        self.state['audits'].append(out)
        return out

    def campaign(self, p):
        dataset = p.get('dataset', 'breast_cancer')
        budget = bounded(p, 'trials', 8, 3, 25, True)
        seed = bounded(p, 'seed', 42, 0, 100000, True)
        if dataset not in ['breast_cancer', 'wine']:
            raise ValueError('Unknown bundled dataset')
        d = load_breast_cancer() if dataset == 'breast_cancer' else load_wine()
        ids = np.arange(len(d.target))
        train, test = train_test_split(ids, test_size=0.2, stratify=d.target, random_state=seed)
        train, val = train_test_split(train, test_size=0.25, stratify=d.target[train], random_state=seed + 1)
        r = np.random.default_rng(seed)
        rows = []
        best = -1
        winner = None
        for i in range(budget):
            self.check_cancel()
            family = ['logistic', 'forest', 'hist_gradient'][i % 3]
            start = time.monotonic()
            config = {'family': family}
            if family == 'logistic':
                c = float(10 ** r.uniform(-2, 2))
                config['C'] = c
                m = LogisticRegression(C=c, max_iter=900, random_state=seed)
            elif family == 'forest':
                depth = int(r.choice([3, 5, 8, 12]))
                config['max_depth'] = depth
                m = RandomForestClassifier(n_estimators=80, max_depth=depth, random_state=seed, n_jobs=1)
            else:
                rate = float(r.choice([0.03, 0.08, 0.15]))
                config['learning_rate'] = rate
                m = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=12, learning_rate=rate, random_state=seed)
            pipeline = make_pipeline(SimpleImputer(), StandardScaler(), m)
            pipeline.fit(d.data[train], d.target[train])
            pred = pipeline.predict(d.data[val])
            score = float(f1_score(d.target[val], pred, average='macro'))
            row = {'trial': i + 1, 'config': config, 'validation_macro_f1': score, 'seconds': time.monotonic() - start, 'status': 'succeeded'}
            rows.append(row)
            if score > best:
                best = score
                winner = pipeline
                winning = dict(row)
        self.check_cancel()
        test_pred = winner.predict(d.data[test])
        result = {'dataset': dataset, 'trials': rows, 'winner': winning, 'test_accuracy': float(accuracy_score(d.target[test], test_pred)), 'test_macro_f1': float(f1_score(d.target[test], test_pred, average='macro')), 'split': {'train': train.tolist(), 'validation': val.tolist(), 'test': test.tolist()}, 'seed': seed, 'note': 'Fixed stratified split and allowlisted budgeted random search. Repeated campaigns can overfit the published benchmark; test values do not feed trial selection.'}
        import joblib
        (RUNTIME / 'models').mkdir(parents=True, exist_ok=True)
        joblib.dump(winner, RUNTIME / 'models' / 'automl-winner.joblib')
        self.state.update(dataset=dataset, trials=rows, last_result=result)
        return result

    def evaluate(self, p):
        return self.campaign(p)
