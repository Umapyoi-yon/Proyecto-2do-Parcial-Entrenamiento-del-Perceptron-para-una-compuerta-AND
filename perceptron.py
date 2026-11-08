"""perceptron.py
Lógica pura del Perceptrón Monocapa (SLP) y almacenamiento del historial.
No depende de Streamlit ni de ninguna librería de presentación.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Sequence, Tuple

# Compuerta AND: (x1, x2, t)
AND_DATASET: List[Tuple[int, int, int]] = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1),
]

_ND = 10  # decimales internos para eliminar ruido de coma flotante

"""Redondea a 10 decimales para evitar artefactos como 0.30000000000000004."""

def _r(v: float) -> float:
    return round(float(v) + 0.0, _ND)

"""Función de activación tipo escalón: 1 si z >= 0, 0 en caso contrario."""
def step(z: float) -> int:
    
    return 1 if z >= 0 else 0

"""Registro completo de la evaluación de una muestra en una época."""
@dataclass
class StepRecord:
    
    epoch: int
    sample: int          # índice de la muestra (1..4)
    x1: float
    x2: float
    t: int               # salida deseada
    w1_prev: float
    w2_prev: float
    b_prev: float
    z: float             # suma ponderada
    y: int               # salida del escalón
    e: int               # error = t - y
    w1_new: float
    w2_new: float
    b_new: float

    def as_dict(self) -> dict:
        return asdict(self)

"""Resumen de una época."""
@dataclass
class EpochRecord:
    
    epoch: int
    w1: float
    w2: float
    b: float
    errors: int

    def as_dict(self) -> dict:
        return asdict(self)

"""Perceptrón de 2 entradas con función escalón y regla delta."""
class Perceptron:
    

    def __init__(self, w1: float = 0.0, w2: float = 0.0, b: float = 0.0,
                 eta: float = 0.1, max_epochs: int = 20):
        if eta <= 0:
            raise ValueError("La tasa de aprendizaje debe ser positiva.")
        if max_epochs < 1:
            raise ValueError("El máximo de épocas debe ser al menos 1.")
        self.w1_init, self.w2_init, self.b_init = float(w1), float(w2), float(b)
        self.w1, self.w2, self.b = float(w1), float(w2), float(b)
        self.eta = float(eta)
        self.max_epochs = int(max_epochs)
        self.steps: List[StepRecord] = []
        self.epochs: List[EpochRecord] = []
        self.converged = False

    # ---------------------------------------------------------------- inferencia
    def weighted_sum(self, x1: float, x2: float) -> float:
        return _r(x1 * self.w1 + x2 * self.w2 + self.b)

    def predict(self, x1: float, x2: float) -> Tuple[float, int]:
        """Devuelve (z, y) para la entrada dada con los pesos actuales."""
        z = self.weighted_sum(x1, x2)
        return z, step(z)

    """Entrena hasta convergencia (0 errores en una época) o hasta max_epochs."""
    # ------------------------------------------------------------- entrenamiento
    def fit(self, data: Sequence[Tuple[float, float, int]] = AND_DATASET) -> "Perceptron":
        
        self.w1, self.w2, self.b = self.w1_init, self.w2_init, self.b_init
        self.steps, self.epochs, self.converged = [], [], False

        for epoch in range(1, self.max_epochs + 1):
            errors = 0
            for i, (x1, x2, t) in enumerate(data, start=1):
                w1p, w2p, bp = self.w1, self.w2, self.b
                z = _r(x1 * w1p + x2 * w2p + bp)
                y = step(z)
                e = t - y
                if e != 0:
                    errors += 1
                self.w1 = _r(w1p + self.eta * e * x1)
                self.w2 = _r(w2p + self.eta * e * x2)
                self.b = _r(bp + self.eta * e)  # para el sesgo, x = 1
                self.steps.append(StepRecord(
                    epoch, i, x1, x2, t, _r(w1p), _r(w2p), _r(bp),
                    z, y, e, self.w1, self.w2, self.b))
            self.epochs.append(EpochRecord(epoch, self.w1, self.w2, self.b, errors))
            if errors == 0:
                self.converged = True
                break
        return self

    # ------------------------------------------------------------------ consultas
    def steps_of_epoch(self, epoch: int) -> List[StepRecord]:
        return [s for s in self.steps if s.epoch == epoch]

    def epoch_table(self) -> List[dict]:
        return [e.as_dict() for e in self.epochs]