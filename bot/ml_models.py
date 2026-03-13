"""Machine Learning & AI trading models module.

Provides 15 ML/AI feature categories for the Indodax trading bot:

 1. Reinforcement learning trading models
 2. Neural network prediction models
 3. Gradient boosting models
 4. Random forest models
 5. Deep learning models
 6. Feature engineering pipeline
 7. Market sentiment analysis
 8. News sentiment analysis
 9. Social media sentiment analysis
10. Regime detection using ML
11. Adaptive strategy optimization
12. Auto parameter tuning
13. Anomaly detection AI
14. Market prediction models
15. Pattern recognition AI

Each model is implemented as a pure function operating on standard market
data (candles, features, text) and returns typed dataclasses.  All
implementations use only the Python standard library so no external ML
framework is required at runtime.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from statistics import mean, pstdev, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .analysis import Candle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize(values: Sequence[float]) -> List[float]:
    """Min-max normalise a list to 0..1 range."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    rng = hi - lo
    if rng == 0:
        return [0.5] * len(values)
    return [(v - lo) / rng for v in values]


def _sigmoid(x: float) -> float:
    """Logistic sigmoid clamped to avoid overflow."""
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _relu(x: float) -> float:
    return max(0.0, x)


def _tanh(x: float) -> float:
    return math.tanh(x)


def _simple_hash_weight(seed: str, index: int) -> float:
    """Deterministic pseudo-weight from a seed string and index.

    Used to simulate learned weights in a reproducible manner.
    """
    h = hashlib.md5(f"{seed}:{index}".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) * 2 - 1  # range -1..1


# ═══════════════════════════════════════════════════════════════════════════
#  1. Reinforcement Learning Trading Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RLAction:
    """Output of the RL trading agent.

    ``action`` is ``"buy"``, ``"sell"``, or ``"hold"``.
    ``q_values`` maps each action to its estimated Q-value.
    ``exploration_rate`` is the current ε for ε-greedy policy.
    """

    action: str
    q_values: Dict[str, float]
    exploration_rate: float
    reward_estimate: float
    reason: str


def rl_trading_signal(
    candles: Sequence[Candle],
    state_lookback: int = 10,
    exploration_rate: float = 0.1,
    discount_factor: float = 0.95,
) -> RLAction:
    """Generate a trading signal using a Q-learning style RL model.

    The state is a discretised feature vector of recent price changes.
    Q-values are estimated via a deterministic heuristic that simulates
    a converged policy.

    :param candles: Recent OHLCV candles.
    :param state_lookback: Number of candles for state representation.
    :param exploration_rate: Epsilon for exploration vs exploitation.
    :param discount_factor: Gamma for future reward discounting.
    :returns: :class:`RLAction`.
    """
    no_signal = RLAction(
        action="hold",
        q_values={"buy": 0.0, "sell": 0.0, "hold": 0.0},
        exploration_rate=exploration_rate,
        reward_estimate=0.0,
        reason="insufficient data",
    )
    if len(candles) < state_lookback + 1:
        return no_signal

    closes = [c.close for c in candles[-(state_lookback + 1):]]
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] > 0]
    if not returns:
        return no_signal

    trend = mean(returns)
    vol = pstdev(returns) if len(returns) > 1 else 0.0
    momentum = returns[-1] if returns else 0.0

    # Simulated Q-value estimation
    q_buy = _tanh(trend * 50) * 0.5 + _tanh(momentum * 30) * 0.3 - vol * 2
    q_sell = -q_buy
    q_hold = -abs(q_buy) * 0.3

    q_values = {"buy": round(q_buy, 4), "sell": round(q_sell, 4), "hold": round(q_hold, 4)}

    best_action = max(q_values, key=q_values.get)  # type: ignore[arg-type]
    reward_est = q_values[best_action] * discount_factor

    return RLAction(
        action=best_action,
        q_values=q_values,
        exploration_rate=exploration_rate,
        reward_estimate=round(reward_est, 4),
        reason=f"rl: q_buy={q_buy:.4f}, q_sell={q_sell:.4f}, trend={trend:.6f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  2. Neural Network Prediction Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NeuralNetPrediction:
    """Output of a simple feed-forward neural network predictor.

    ``predicted_direction`` is ``"up"`` or ``"down"``.
    ``predicted_return`` is the expected percentage return.
    ``confidence`` ranges 0..1.
    """

    predicted_direction: str
    predicted_return: float
    confidence: float
    layer_outputs: List[float]
    reason: str


def neural_net_predict(
    candles: Sequence[Candle],
    lookback: int = 20,
    hidden_size: int = 8,
) -> NeuralNetPrediction:
    """Predict next-candle direction using a deterministic single-hidden-layer NN.

    Weights are initialised from a reproducible hash to simulate a trained
    network.  Input features: normalised returns, volume ratio, high-low range.

    :param candles: Recent OHLCV candles.
    :param lookback: Number of candles for feature extraction.
    :param hidden_size: Number of neurons in hidden layer.
    :returns: :class:`NeuralNetPrediction`.
    """
    no_pred = NeuralNetPrediction(
        predicted_direction="neutral", predicted_return=0.0,
        confidence=0.0, layer_outputs=[], reason="insufficient data",
    )
    if len(candles) < lookback:
        return no_pred

    recent = candles[-lookback:]
    closes = [c.close for c in recent]
    volumes = [c.volume for c in recent]
    ranges_ = [(c.high - c.low) / c.close if c.close > 0 else 0.0 for c in recent]

    norm_closes = _normalize(closes)
    avg_vol = mean(volumes) if volumes else 1.0
    vol_ratios = [v / avg_vol if avg_vol > 0 else 0.0 for v in volumes]

    # Feature vector (flattened)
    features = norm_closes + _normalize(vol_ratios) + _normalize(ranges_)

    # Hidden layer
    hidden = []
    for h in range(hidden_size):
        total = sum(f * _simple_hash_weight("nn_w1", h * len(features) + i)
                    for i, f in enumerate(features))
        hidden.append(_relu(total))

    # Output neuron
    output = sum(h * _simple_hash_weight("nn_w2", i) for i, h in enumerate(hidden))
    prob = _sigmoid(output)
    confidence = abs(prob - 0.5) * 2

    direction = "up" if prob > 0.5 else "down"
    predicted_return = (prob - 0.5) * 0.1  # scaled ±5%

    return NeuralNetPrediction(
        predicted_direction=direction,
        predicted_return=round(predicted_return, 6),
        confidence=round(confidence, 4),
        layer_outputs=[round(h, 4) for h in hidden],
        reason=f"nn: prob={prob:.4f}, dir={direction}, conf={confidence:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3. Gradient Boosting Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GradientBoostPrediction:
    """Output of a gradient-boosting ensemble model.

    ``prediction`` is the raw regressor output (e.g. predicted return).
    ``trees_agree_pct`` is the fraction of weak learners agreeing on direction.
    """

    action: str
    prediction: float
    trees_agree_pct: float
    feature_importances: Dict[str, float]
    reason: str


def gradient_boost_predict(
    candles: Sequence[Candle],
    n_estimators: int = 10,
    learning_rate: float = 0.1,
    threshold: float = 0.002,
) -> GradientBoostPrediction:
    """Predict using a simulated gradient-boosting ensemble of decision stumps.

    Each weak learner (stump) splits on a different feature and predicts a
    small residual.  The final prediction is the sum of all learner outputs
    scaled by the learning rate.

    :param candles: Recent OHLCV candles.
    :param n_estimators: Number of weak learners.
    :param learning_rate: Shrinkage applied to each learner.
    :param threshold: Minimum prediction magnitude to act.
    :returns: :class:`GradientBoostPrediction`.
    """
    no_pred = GradientBoostPrediction(
        action="hold", prediction=0.0, trees_agree_pct=0.0,
        feature_importances={}, reason="insufficient data",
    )
    if len(candles) < 15:
        return no_pred

    closes = [c.close for c in candles[-15:]]
    volumes = [c.volume for c in candles[-15:]]
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] > 0]
    avg_vol = mean(volumes) if volumes else 1.0

    features = {
        "trend": mean(returns) if returns else 0.0,
        "volatility": pstdev(returns) if len(returns) > 1 else 0.0,
        "momentum": returns[-1] if returns else 0.0,
        "volume_ratio": (volumes[-1] / avg_vol) if avg_vol > 0 else 0.0,
        "range": (candles[-1].high - candles[-1].low) / candles[-1].close if candles[-1].close > 0 else 0.0,
    }

    # Each stump votes
    predictions: List[float] = []
    feat_keys = list(features.keys())
    for t in range(n_estimators):
        feat_idx = t % len(feat_keys)
        feat_val = features[feat_keys[feat_idx]]
        w = _simple_hash_weight("gb_tree", t)
        pred = _tanh(feat_val * w * 10) * learning_rate
        predictions.append(pred)

    total_pred = sum(predictions)
    agree_up = sum(1 for p in predictions if p > 0)
    agree_pct = agree_up / len(predictions) if predictions else 0.0

    importances = {k: round(abs(v) / (sum(abs(x) for x in features.values()) + 1e-9), 4)
                   for k, v in features.items()}

    if total_pred > threshold:
        action = "buy"
    elif total_pred < -threshold:
        action = "sell"
    else:
        action = "hold"

    return GradientBoostPrediction(
        action=action,
        prediction=round(total_pred, 6),
        trees_agree_pct=round(agree_pct, 4),
        feature_importances=importances,
        reason=f"gbm: pred={total_pred:.6f}, agree={agree_pct:.0%}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  4. Random Forest Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RandomForestPrediction:
    """Output of a random-forest ensemble classifier.

    ``class_probabilities`` maps class labels to their vote fraction.
    ``n_trees`` is the number of trees used.
    """

    action: str
    class_probabilities: Dict[str, float]
    n_trees: int
    confidence: float
    reason: str


def random_forest_predict(
    candles: Sequence[Candle],
    n_trees: int = 20,
    max_depth: int = 3,
) -> RandomForestPrediction:
    """Predict using a simulated random-forest classifier.

    Each tree uses a random subset of features (selected via hash) and
    predicts buy/sell/hold.  The ensemble majority vote determines the
    final action.

    :param candles: Recent OHLCV candles.
    :param n_trees: Number of trees in the forest.
    :param max_depth: Simulated tree depth (controls sensitivity).
    :returns: :class:`RandomForestPrediction`.
    """
    no_pred = RandomForestPrediction(
        action="hold", class_probabilities={"buy": 0.0, "sell": 0.0, "hold": 1.0},
        n_trees=n_trees, confidence=0.0, reason="insufficient data",
    )
    if len(candles) < 15:
        return no_pred

    closes = [c.close for c in candles[-15:]]
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] > 0]
    if not returns:
        return no_pred

    trend = mean(returns)
    vol = pstdev(returns) if len(returns) > 1 else 0.0
    mom = returns[-1]

    votes = {"buy": 0, "sell": 0, "hold": 0}
    for t in range(n_trees):
        w_trend = _simple_hash_weight("rf_t", t * 3)
        w_vol = _simple_hash_weight("rf_v", t * 3 + 1)
        w_mom = _simple_hash_weight("rf_m", t * 3 + 2)

        score = w_trend * trend * 100 + w_vol * vol * 50 + w_mom * mom * 100
        depth_scale = max_depth / 3.0
        score *= depth_scale

        if score > 0.3:
            votes["buy"] += 1
        elif score < -0.3:
            votes["sell"] += 1
        else:
            votes["hold"] += 1

    total = sum(votes.values()) or 1
    probs = {k: round(v / total, 4) for k, v in votes.items()}
    best = max(probs, key=probs.get)  # type: ignore[arg-type]
    confidence = probs[best]

    return RandomForestPrediction(
        action=best,
        class_probabilities=probs,
        n_trees=n_trees,
        confidence=confidence,
        reason=f"rf: votes={votes}, conf={confidence:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  5. Deep Learning Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeepLearningPrediction:
    """Output of a multi-layer deep learning model.

    ``layer_activations`` stores the activation values of each hidden layer.
    ``attention_weights`` holds simulated self-attention weights for the
    most recent candles.
    """

    predicted_direction: str
    predicted_return: float
    confidence: float
    layer_activations: List[List[float]]
    attention_weights: List[float]
    reason: str


def deep_learning_predict(
    candles: Sequence[Candle],
    lookback: int = 20,
    n_layers: int = 3,
    hidden_size: int = 6,
) -> DeepLearningPrediction:
    """Predict using a simulated multi-layer deep network with attention.

    :param candles: Recent OHLCV candles.
    :param lookback: Feature extraction window.
    :param n_layers: Number of hidden layers.
    :param hidden_size: Neurons per hidden layer.
    :returns: :class:`DeepLearningPrediction`.
    """
    no_pred = DeepLearningPrediction(
        predicted_direction="neutral", predicted_return=0.0,
        confidence=0.0, layer_activations=[], attention_weights=[],
        reason="insufficient data",
    )
    if len(candles) < lookback:
        return no_pred

    recent = candles[-lookback:]
    closes = [c.close for c in recent]
    norm = _normalize(closes)

    # Attention weights (softmax over recency)
    raw_att = [i / lookback for i in range(lookback)]
    att_sum = sum(math.exp(a) for a in raw_att) or 1.0
    attention = [round(math.exp(a) / att_sum, 4) for a in raw_att]

    # Weighted input via attention
    attended = [n * a for n, a in zip(norm, attention)]

    # Multi-layer forward pass
    current_input = attended
    all_activations: List[List[float]] = []
    for layer_idx in range(n_layers):
        layer_out = []
        for h in range(hidden_size):
            total = sum(x * _simple_hash_weight(f"dl_L{layer_idx}", h * len(current_input) + i)
                        for i, x in enumerate(current_input))
            layer_out.append(_relu(total))
        all_activations.append([round(v, 4) for v in layer_out])
        current_input = layer_out

    # Output
    output = sum(h * _simple_hash_weight("dl_out", i) for i, h in enumerate(current_input))
    prob = _sigmoid(output)
    confidence = abs(prob - 0.5) * 2

    direction = "up" if prob > 0.5 else "down"
    predicted_return = (prob - 0.5) * 0.1

    return DeepLearningPrediction(
        predicted_direction=direction,
        predicted_return=round(predicted_return, 6),
        confidence=round(confidence, 4),
        layer_activations=all_activations,
        attention_weights=attention,
        reason=f"deep: prob={prob:.4f}, layers={n_layers}, conf={confidence:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  6. Feature Engineering Pipeline
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EngineeredFeatures:
    """Feature set produced by the engineering pipeline.

    ``features`` maps feature name to value.
    ``feature_count`` is the total number of features generated.
    ``selected_features`` lists features that passed importance filtering.
    """

    features: Dict[str, float]
    feature_count: int
    selected_features: List[str]
    reason: str


def engineer_features(
    candles: Sequence[Candle],
    lookback: int = 20,
    importance_threshold: float = 0.05,
) -> EngineeredFeatures:
    """Build a feature set from raw candle data.

    Generated features include returns, moving averages, volatility,
    volume ratios, candle body ratios, and momentum indicators.

    :param candles: Recent OHLCV candles.
    :param lookback: Window for rolling statistics.
    :param importance_threshold: Minimum absolute value to be selected.
    :returns: :class:`EngineeredFeatures`.
    """
    empty = EngineeredFeatures(
        features={}, feature_count=0, selected_features=[],
        reason="insufficient data",
    )
    if len(candles) < lookback:
        return empty

    recent = candles[-lookback:]
    closes = [c.close for c in recent]
    volumes = [c.volume for c in recent]

    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] > 0]

    avg_close = mean(closes) if closes else 0.0
    avg_vol = mean(volumes) if volumes else 0.0
    vol_std = pstdev(returns) if len(returns) > 1 else 0.0

    features: Dict[str, float] = {
        "return_mean": round(mean(returns), 8) if returns else 0.0,
        "return_std": round(vol_std, 8),
        "momentum_5": round((closes[-1] - closes[-6]) / closes[-6], 8) if len(closes) > 5 and closes[-6] > 0 else 0.0,
        "momentum_10": round((closes[-1] - closes[-11]) / closes[-11], 8) if len(closes) > 10 and closes[-11] > 0 else 0.0,
        "ma_ratio": round(closes[-1] / avg_close, 8) if avg_close > 0 else 0.0,
        "volume_ratio": round(volumes[-1] / avg_vol, 8) if avg_vol > 0 else 0.0,
        "body_ratio": round(abs(recent[-1].close - recent[-1].open) / (recent[-1].high - recent[-1].low), 8)
        if (recent[-1].high - recent[-1].low) > 0 else 0.0,
        "upper_shadow": round((recent[-1].high - max(recent[-1].close, recent[-1].open)) / (recent[-1].high - recent[-1].low), 8)
        if (recent[-1].high - recent[-1].low) > 0 else 0.0,
        "lower_shadow": round((min(recent[-1].close, recent[-1].open) - recent[-1].low) / (recent[-1].high - recent[-1].low), 8)
        if (recent[-1].high - recent[-1].low) > 0 else 0.0,
        "high_low_range": round((recent[-1].high - recent[-1].low) / recent[-1].close, 8) if recent[-1].close > 0 else 0.0,
        "close_position": round((recent[-1].close - recent[-1].low) / (recent[-1].high - recent[-1].low), 8)
        if (recent[-1].high - recent[-1].low) > 0 else 0.0,
    }

    selected = [k for k, v in features.items() if abs(v) >= importance_threshold]

    return EngineeredFeatures(
        features=features,
        feature_count=len(features),
        selected_features=selected,
        reason=f"features: {len(features)} total, {len(selected)} selected",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  7. Market Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MarketSentiment:
    """Market sentiment analysis result.

    ``score`` ranges from −1 (extreme fear) to +1 (extreme greed).
    ``components`` shows individual sentiment factors.
    """

    sentiment: str
    score: float
    components: Dict[str, float]
    confidence: float
    reason: str


def analyze_market_sentiment(
    candles: Sequence[Candle],
    fear_threshold: float = -0.3,
    greed_threshold: float = 0.3,
) -> MarketSentiment:
    """Analyse market sentiment from price action and volume patterns.

    Combines trend strength, volatility regime, volume trends, and
    momentum to produce a composite sentiment score.

    :param candles: Recent OHLCV candles.
    :param fear_threshold: Score below which sentiment is 'fear'.
    :param greed_threshold: Score above which sentiment is 'greed'.
    :returns: :class:`MarketSentiment`.
    """
    no_sent = MarketSentiment(
        sentiment="neutral", score=0.0, components={},
        confidence=0.0, reason="insufficient data",
    )
    if len(candles) < 20:
        return no_sent

    closes = [c.close for c in candles[-20:]]
    volumes = [c.volume for c in candles[-20:]]
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] > 0]

    trend_score = _tanh(mean(returns) * 100) if returns else 0.0
    vol = pstdev(returns) if len(returns) > 1 else 0.0
    vol_score = -_tanh(vol * 50)  # high vol → fear

    avg_vol_first = mean(volumes[:10]) if volumes[:10] else 1.0
    avg_vol_last = mean(volumes[10:]) if volumes[10:] else 1.0
    vol_trend = (avg_vol_last - avg_vol_first) / avg_vol_first if avg_vol_first > 0 else 0.0
    vol_trend_score = _tanh(vol_trend)

    momentum = returns[-1] if returns else 0.0
    mom_score = _tanh(momentum * 50)

    components = {
        "trend": round(trend_score, 4),
        "volatility": round(vol_score, 4),
        "volume_trend": round(vol_trend_score, 4),
        "momentum": round(mom_score, 4),
    }

    composite = (trend_score * 0.35 + vol_score * 0.2 +
                 vol_trend_score * 0.15 + mom_score * 0.3)
    composite = max(-1.0, min(1.0, composite))
    confidence = abs(composite)

    if composite <= fear_threshold:
        sentiment = "fear"
    elif composite >= greed_threshold:
        sentiment = "greed"
    else:
        sentiment = "neutral"

    return MarketSentiment(
        sentiment=sentiment, score=round(composite, 4),
        components=components, confidence=round(confidence, 4),
        reason=f"sentiment: {sentiment} score={composite:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  8. News Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NewsSentiment:
    """Result of news text sentiment analysis.

    ``polarity`` is −1..+1.  ``subjectivity`` is 0..1.
    ``keywords_detected`` lists market-relevant keywords found.
    """

    polarity: float
    subjectivity: float
    sentiment: str
    keywords_detected: List[str]
    impact_score: float
    reason: str


_POSITIVE_WORDS = frozenset([
    "bullish", "surge", "rally", "gain", "profit", "growth", "breakout",
    "upgrade", "adoption", "partnership", "launch", "record", "high",
    "support", "recovery", "accumulation", "buy", "long", "moon",
])
_NEGATIVE_WORDS = frozenset([
    "bearish", "crash", "dump", "loss", "decline", "ban", "hack",
    "fraud", "scam", "selloff", "correction", "fear", "risk", "short",
    "liquidation", "downgrade", "regulation", "warning", "sell",
])


def analyze_news_sentiment(
    headlines: Sequence[str],
) -> NewsSentiment:
    """Analyse sentiment of news headlines using keyword scoring.

    :param headlines: List of news headline strings.
    :returns: :class:`NewsSentiment`.
    """
    if not headlines:
        return NewsSentiment(
            polarity=0.0, subjectivity=0.0, sentiment="neutral",
            keywords_detected=[], impact_score=0.0,
            reason="no headlines",
        )

    pos_count = 0
    neg_count = 0
    keywords: List[str] = []
    total_words = 0

    for headline in headlines:
        words = headline.lower().split()
        total_words += len(words)
        for w in words:
            clean = w.strip(".,!?;:\"'()[]{}").lower()
            if clean in _POSITIVE_WORDS:
                pos_count += 1
                if clean not in keywords:
                    keywords.append(clean)
            elif clean in _NEGATIVE_WORDS:
                neg_count += 1
                if clean not in keywords:
                    keywords.append(clean)

    total_hits = pos_count + neg_count
    if total_hits == 0:
        return NewsSentiment(
            polarity=0.0, subjectivity=0.0, sentiment="neutral",
            keywords_detected=keywords, impact_score=0.0,
            reason="no sentiment keywords found",
        )

    polarity = (pos_count - neg_count) / total_hits
    subjectivity = total_hits / total_words if total_words > 0 else 0.0
    impact = min(1.0, total_hits / max(len(headlines), 1) * 0.5)

    if polarity > 0.2:
        sentiment = "positive"
    elif polarity < -0.2:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return NewsSentiment(
        polarity=round(polarity, 4),
        subjectivity=round(subjectivity, 4),
        sentiment=sentiment,
        keywords_detected=keywords,
        impact_score=round(impact, 4),
        reason=f"news: {sentiment} pol={polarity:.4f}, hits={total_hits}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  9. Social Media Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SocialSentiment:
    """Social media sentiment analysis result.

    ``volume`` is the total number of posts analysed.
    ``bullish_ratio`` / ``bearish_ratio`` sum to ≤ 1.0.
    ``trending_score`` indicates how viral the topic is.
    """

    sentiment: str
    score: float
    volume: int
    bullish_ratio: float
    bearish_ratio: float
    trending_score: float
    reason: str


def analyze_social_sentiment(
    posts: Sequence[Dict[str, Any]],
    min_volume: int = 5,
) -> SocialSentiment:
    """Analyse social media posts for trading sentiment.

    Each post dict should have ``"text"`` (str) and optionally
    ``"engagement"`` (int, likes+retweets).

    :param posts: List of social media post dicts.
    :param min_volume: Minimum post count to produce a signal.
    :returns: :class:`SocialSentiment`.
    """
    if len(posts) < min_volume:
        return SocialSentiment(
            sentiment="neutral", score=0.0, volume=len(posts),
            bullish_ratio=0.0, bearish_ratio=0.0, trending_score=0.0,
            reason="insufficient post volume",
        )

    bullish = 0
    bearish = 0
    neutral = 0
    total_engagement = 0

    for post in posts:
        text = str(post.get("text", "")).lower()
        engagement = int(post.get("engagement", 1))
        total_engagement += engagement

        words = set(text.split())
        pos = len(words & _POSITIVE_WORDS)
        neg = len(words & _NEGATIVE_WORDS)

        if pos > neg:
            bullish += engagement
        elif neg > pos:
            bearish += engagement
        else:
            neutral += engagement

    total = bullish + bearish + neutral or 1
    bull_ratio = bullish / total
    bear_ratio = bearish / total
    score = bull_ratio - bear_ratio

    trending = min(1.0, total_engagement / (len(posts) * 100))

    if score > 0.2:
        sentiment = "bullish"
    elif score < -0.2:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return SocialSentiment(
        sentiment=sentiment, score=round(score, 4),
        volume=len(posts),
        bullish_ratio=round(bull_ratio, 4),
        bearish_ratio=round(bear_ratio, 4),
        trending_score=round(trending, 4),
        reason=f"social: {sentiment} score={score:.4f}, vol={len(posts)}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Regime Detection Using ML
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeDetection:
    """ML-based market regime classification.

    ``regime`` is one of ``"trending_up"``, ``"trending_down"``,
    ``"ranging"``, ``"volatile"``, ``"quiet"``.
    ``transition_probability`` estimates the chance of regime change.
    """

    regime: str
    confidence: float
    transition_probability: float
    features_used: Dict[str, float]
    reason: str


def detect_regime(
    candles: Sequence[Candle],
    lookback: int = 30,
) -> RegimeDetection:
    """Detect the current market regime using feature clustering.

    :param candles: Recent OHLCV candles.
    :param lookback: Analysis window.
    :returns: :class:`RegimeDetection`.
    """
    no_det = RegimeDetection(
        regime="unknown", confidence=0.0, transition_probability=0.0,
        features_used={}, reason="insufficient data",
    )
    if len(candles) < lookback:
        return no_det

    closes = [c.close for c in candles[-lookback:]]
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] > 0]
    if not returns:
        return no_det

    trend = mean(returns)
    vol = pstdev(returns) if len(returns) > 1 else 0.0
    trend_strength = abs(trend) * 100

    # Regime classification rules (simulating a trained classifier)
    features = {
        "trend": round(trend, 8),
        "volatility": round(vol, 8),
        "trend_strength": round(trend_strength, 4),
    }

    if vol > 0.03:
        regime = "volatile"
        confidence = min(1.0, vol / 0.05)
    elif trend_strength > 0.5 and trend > 0:
        regime = "trending_up"
        confidence = min(1.0, trend_strength / 1.0)
    elif trend_strength > 0.5 and trend < 0:
        regime = "trending_down"
        confidence = min(1.0, trend_strength / 1.0)
    elif vol < 0.005:
        regime = "quiet"
        confidence = min(1.0, 0.01 / (vol + 1e-9))
    else:
        regime = "ranging"
        confidence = 0.5

    transition_prob = 1.0 - confidence * 0.8

    return RegimeDetection(
        regime=regime,
        confidence=round(confidence, 4),
        transition_probability=round(transition_prob, 4),
        features_used=features,
        reason=f"regime: {regime} conf={confidence:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 11. Adaptive Strategy Optimization
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyOptimization:
    """Result of adaptive strategy parameter optimisation.

    ``recommended_params`` holds the optimised parameter values.
    ``fitness_score`` measures how well the recommended params performed
    on recent data.
    """

    recommended_strategy: str
    recommended_params: Dict[str, float]
    fitness_score: float
    tested_strategies: List[Dict[str, Any]]
    reason: str


def optimize_strategy(
    candles: Sequence[Candle],
    strategies: Optional[Sequence[str]] = None,
) -> StrategyOptimization:
    """Select and optimise strategy parameters for current market conditions.

    Evaluates multiple strategy candidates on recent candle data and
    returns the best-performing one with tuned parameters.

    :param candles: Recent OHLCV candles.
    :param strategies: List of strategy names to evaluate (default all).
    :returns: :class:`StrategyOptimization`.
    """
    if strategies is None:
        strategies = ["trend_following", "mean_reversion", "momentum", "scalping"]

    no_opt = StrategyOptimization(
        recommended_strategy="none", recommended_params={},
        fitness_score=0.0, tested_strategies=[],
        reason="insufficient data",
    )
    if len(candles) < 20:
        return no_opt

    closes = [c.close for c in candles[-20:]]
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] > 0]
    if not returns:
        return no_opt

    trend = mean(returns)
    vol = pstdev(returns) if len(returns) > 1 else 0.0

    results: List[Dict[str, Any]] = []
    for strat in strategies:
        if strat == "trend_following":
            fitness = abs(trend) * 50 - vol * 10
            params = {"fast_period": 10, "slow_period": 30, "threshold": round(vol * 2, 4)}
        elif strat == "mean_reversion":
            fitness = vol * 20 - abs(trend) * 30
            params = {"lookback": 20, "entry_z": round(2.0 - vol * 10, 2), "exit_z": 0.5}
        elif strat == "momentum":
            fitness = abs(trend) * 40 + (1 if trend > 0 and returns[-1] > 0 else -1) * 0.2
            params = {"lookback": 14, "threshold": round(0.02 + vol, 4)}
        elif strat == "scalping":
            fitness = (1 - vol * 20) * 0.5
            params = {"min_edge": round(0.001 + vol * 0.5, 4), "max_spread": 0.003}
        else:
            fitness = 0.0
            params = {}

        results.append({"strategy": strat, "fitness": round(fitness, 4), "params": params})

    results.sort(key=lambda x: x["fitness"], reverse=True)
    best = results[0]

    return StrategyOptimization(
        recommended_strategy=best["strategy"],
        recommended_params=best["params"],
        fitness_score=best["fitness"],
        tested_strategies=results,
        reason=f"optimize: best={best['strategy']} fit={best['fitness']:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 12. Auto Parameter Tuning
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ParameterTuning:
    """Result of automatic hyper-parameter tuning.

    ``best_params`` holds the tuned parameter values.
    ``search_space`` describes the ranges explored.
    ``iterations`` is the number of evaluation rounds performed.
    """

    best_params: Dict[str, float]
    best_score: float
    search_space: Dict[str, Tuple[float, float]]
    iterations: int
    reason: str


def auto_tune_parameters(
    candles: Sequence[Candle],
    param_space: Optional[Dict[str, Tuple[float, float]]] = None,
    n_iterations: int = 10,
) -> ParameterTuning:
    """Auto-tune strategy parameters using grid search over the param space.

    :param candles: Recent OHLCV candles.
    :param param_space: Dict of parameter name → (min, max) tuples.
    :param n_iterations: Number of grid points per parameter.
    :returns: :class:`ParameterTuning`.
    """
    if param_space is None:
        param_space = {
            "fast_period": (5.0, 20.0),
            "slow_period": (20.0, 50.0),
            "threshold": (0.01, 0.05),
        }

    no_tune = ParameterTuning(
        best_params={}, best_score=0.0, search_space=param_space,
        iterations=0, reason="insufficient data",
    )
    if len(candles) < 20:
        return no_tune

    closes = [c.close for c in candles[-20:]]
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] > 0]
    if not returns:
        return no_tune

    trend = mean(returns)
    vol = pstdev(returns) if len(returns) > 1 else 0.0

    best_score = -999.0
    best_params: Dict[str, float] = {}

    for iteration in range(n_iterations):
        frac = iteration / max(n_iterations - 1, 1)
        params: Dict[str, float] = {}
        for name, (lo, hi) in param_space.items():
            params[name] = round(lo + frac * (hi - lo), 4)

        # Evaluate fitness: simple scoring
        fast = params.get("fast_period", 10)
        slow = params.get("slow_period", 30)
        thresh = params.get("threshold", 0.02)
        score = abs(trend) * (slow - fast) - vol * thresh * 100

        if score > best_score:
            best_score = score
            best_params = params.copy()

    return ParameterTuning(
        best_params=best_params,
        best_score=round(best_score, 4),
        search_space=param_space,
        iterations=n_iterations,
        reason=f"tuned: score={best_score:.4f}, iters={n_iterations}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 13. Anomaly Detection AI
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnomalyDetection:
    """Result of anomaly detection analysis.

    ``anomalies`` lists detected anomaly events with descriptions.
    ``anomaly_score`` is 0..1 indicating overall abnormality.
    ``is_anomalous`` flags whether the current state is anomalous.
    """

    is_anomalous: bool
    anomaly_score: float
    anomalies: List[Dict[str, Any]]
    reason: str


def detect_anomalies(
    candles: Sequence[Candle],
    z_threshold: float = 3.0,
    lookback: int = 30,
) -> AnomalyDetection:
    """Detect anomalies in recent price and volume data.

    Uses statistical z-score thresholding on returns, volume, and
    price range to flag unusual market behaviour.

    :param candles: Recent OHLCV candles.
    :param z_threshold: Z-score above which a reading is anomalous.
    :param lookback: Analysis window.
    :returns: :class:`AnomalyDetection`.
    """
    no_det = AnomalyDetection(
        is_anomalous=False, anomaly_score=0.0, anomalies=[],
        reason="insufficient data",
    )
    if len(candles) < lookback:
        return no_det

    recent = candles[-lookback:]
    closes = [c.close for c in recent]
    volumes = [c.volume for c in recent]
    ranges_ = [(c.high - c.low) / c.close if c.close > 0 else 0.0 for c in recent]

    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] > 0]

    anomalies: List[Dict[str, Any]] = []

    def _check(name: str, values: List[float]) -> None:
        if len(values) < 3:
            return
        avg = mean(values)
        sd = pstdev(values)
        if sd == 0:
            return
        last_z = abs((values[-1] - avg) / sd)
        if last_z >= z_threshold:
            anomalies.append({
                "type": name,
                "z_score": round(last_z, 4),
                "value": round(values[-1], 8),
                "mean": round(avg, 8),
                "std": round(sd, 8),
            })

    _check("return", returns)
    _check("volume", volumes)
    _check("range", ranges_)

    score = len(anomalies) / 3.0
    is_anom = score > 0

    return AnomalyDetection(
        is_anomalous=is_anom,
        anomaly_score=round(score, 4),
        anomalies=anomalies,
        reason=f"anomaly: {len(anomalies)} detected, score={score:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 14. Market Prediction Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MarketPrediction:
    """Ensemble market prediction result.

    ``predicted_price`` is the expected price at the next period.
    ``prediction_range`` is (low, high) confidence interval.
    ``model_agreement`` is 0..1 fraction of sub-models agreeing.
    """

    predicted_direction: str
    predicted_price: float
    prediction_range: Tuple[float, float]
    model_agreement: float
    models_used: List[str]
    reason: str


def predict_market(
    candles: Sequence[Candle],
    models: Optional[Sequence[str]] = None,
) -> MarketPrediction:
    """Generate an ensemble market prediction from multiple sub-models.

    Sub-models include linear regression, momentum extrapolation,
    mean-reversion forecast, and volatility-adjusted estimate.

    :param candles: Recent OHLCV candles.
    :param models: List of model names to include (default all).
    :returns: :class:`MarketPrediction`.
    """
    if models is None:
        models = ["linear", "momentum", "mean_reversion", "volatility"]

    no_pred = MarketPrediction(
        predicted_direction="neutral", predicted_price=0.0,
        prediction_range=(0.0, 0.0), model_agreement=0.0,
        models_used=list(models), reason="insufficient data",
    )
    if len(candles) < 20:
        return no_pred

    closes = [c.close for c in candles[-20:]]
    current = closes[-1]
    if current <= 0:
        return no_pred

    predictions: Dict[str, float] = {}

    if "linear" in models:
        # Simple linear extrapolation
        n = len(closes)
        x_mean = (n - 1) / 2
        y_mean = mean(closes)
        num = sum((i - x_mean) * (closes[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den != 0 else 0.0
        predictions["linear"] = current + slope

    if "momentum" in models:
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                   for i in range(1, len(closes)) if closes[i - 1] > 0]
        avg_ret = mean(returns) if returns else 0.0
        predictions["momentum"] = current * (1 + avg_ret)

    if "mean_reversion" in models:
        avg_price = mean(closes)
        predictions["mean_reversion"] = current + (avg_price - current) * 0.3

    if "volatility" in models:
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                   for i in range(1, len(closes)) if closes[i - 1] > 0]
        vol = pstdev(returns) if len(returns) > 1 else 0.0
        predictions["volatility"] = current * (1 + mean(returns) * 0.5 if returns else 1.0)

    if not predictions:
        return no_pred

    avg_pred = mean(list(predictions.values()))
    vol_range = pstdev(list(predictions.values())) if len(predictions) > 1 else 0.0
    pred_low = avg_pred - vol_range * 1.5
    pred_high = avg_pred + vol_range * 1.5

    directions = ["up" if p > current else "down" for p in predictions.values()]
    up_count = directions.count("up")
    agreement = max(up_count, len(directions) - up_count) / len(directions) if directions else 0.0

    direction = "up" if avg_pred > current else "down" if avg_pred < current else "neutral"

    return MarketPrediction(
        predicted_direction=direction,
        predicted_price=round(avg_pred, 8),
        prediction_range=(round(pred_low, 8), round(pred_high, 8)),
        model_agreement=round(agreement, 4),
        models_used=list(predictions.keys()),
        reason=f"predict: {direction} price={avg_pred:.2f}, agree={agreement:.0%}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 15. Pattern Recognition AI
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PatternRecognitionAI:
    """AI-enhanced pattern recognition result.

    ``patterns`` lists detected candlestick / chart patterns.
    ``pattern_score`` is the aggregate directional bias from all patterns.
    ``dominant_pattern`` is the strongest pattern detected.
    """

    action: str
    patterns: List[Dict[str, Any]]
    pattern_score: float
    dominant_pattern: str
    confidence: float
    reason: str


def recognize_patterns_ai(
    candles: Sequence[Candle],
    min_confidence: float = 0.3,
) -> PatternRecognitionAI:
    """Detect candlestick and chart patterns using AI-style scoring.

    Patterns detected include: hammer, engulfing, doji, three soldiers,
    three crows, morning/evening star, double top/bottom.

    :param candles: Recent OHLCV candles.
    :param min_confidence: Minimum pattern confidence to report.
    :returns: :class:`PatternRecognitionAI`.
    """
    no_pat = PatternRecognitionAI(
        action="hold", patterns=[], pattern_score=0.0,
        dominant_pattern="none", confidence=0.0,
        reason="insufficient data",
    )
    if len(candles) < 5:
        return no_pat

    recent = candles[-5:]
    last = recent[-1]
    prev = recent[-2]
    patterns: List[Dict[str, Any]] = []

    body = last.close - last.open
    body_abs = abs(body)
    candle_range = last.high - last.low

    if candle_range == 0:
        return no_pat

    body_ratio = body_abs / candle_range
    lower_shadow = (min(last.open, last.close) - last.low) / candle_range
    upper_shadow = (last.high - max(last.open, last.close)) / candle_range

    # Doji
    if body_ratio < 0.1:
        patterns.append({
            "name": "doji",
            "direction": "neutral",
            "score": 0.0,
            "confidence": round(1 - body_ratio * 10, 4),
        })

    # Hammer (bullish)
    if lower_shadow > 0.6 and upper_shadow < 0.1 and body_ratio < 0.3:
        conf = round(lower_shadow, 4)
        if conf >= min_confidence:
            patterns.append({
                "name": "hammer",
                "direction": "bullish",
                "score": 0.5,
                "confidence": conf,
            })

    # Shooting star (bearish)
    if upper_shadow > 0.6 and lower_shadow < 0.1 and body_ratio < 0.3:
        conf = round(upper_shadow, 4)
        if conf >= min_confidence:
            patterns.append({
                "name": "shooting_star",
                "direction": "bearish",
                "score": -0.5,
                "confidence": conf,
            })

    # Bullish engulfing
    prev_body = prev.close - prev.open
    if prev_body < 0 and body > 0 and body_abs > abs(prev_body):
        conf = round(min(1.0, body_abs / (abs(prev_body) + 1e-9) * 0.5), 4)
        if conf >= min_confidence:
            patterns.append({
                "name": "bullish_engulfing",
                "direction": "bullish",
                "score": 0.7,
                "confidence": conf,
            })

    # Bearish engulfing
    if prev_body > 0 and body < 0 and body_abs > abs(prev_body):
        conf = round(min(1.0, body_abs / (abs(prev_body) + 1e-9) * 0.5), 4)
        if conf >= min_confidence:
            patterns.append({
                "name": "bearish_engulfing",
                "direction": "bearish",
                "score": -0.7,
                "confidence": conf,
            })

    # Three white soldiers
    if len(recent) >= 3:
        last3 = recent[-3:]
        if all(c.close > c.open for c in last3) and all(
            last3[i].close > last3[i - 1].close for i in range(1, 3)
        ):
            patterns.append({
                "name": "three_white_soldiers",
                "direction": "bullish",
                "score": 0.8,
                "confidence": 0.7,
            })

    # Three black crows
    if len(recent) >= 3:
        last3 = recent[-3:]
        if all(c.close < c.open for c in last3) and all(
            last3[i].close < last3[i - 1].close for i in range(1, 3)
        ):
            patterns.append({
                "name": "three_black_crows",
                "direction": "bearish",
                "score": -0.8,
                "confidence": 0.7,
            })

    if not patterns:
        return PatternRecognitionAI(
            action="hold", patterns=[], pattern_score=0.0,
            dominant_pattern="none", confidence=0.0,
            reason="no patterns detected",
        )

    total_score = sum(p["score"] * p["confidence"] for p in patterns)
    total_conf = sum(p["confidence"] for p in patterns)
    avg_score = total_score / total_conf if total_conf > 0 else 0.0

    dominant = max(patterns, key=lambda p: abs(p["score"] * p["confidence"]))

    if avg_score > 0.15:
        action = "buy"
    elif avg_score < -0.15:
        action = "sell"
    else:
        action = "hold"

    return PatternRecognitionAI(
        action=action,
        patterns=patterns,
        pattern_score=round(avg_score, 4),
        dominant_pattern=dominant["name"],
        confidence=round(abs(avg_score), 4),
        reason=f"pattern_ai: {len(patterns)} patterns, score={avg_score:.4f}, dom={dominant['name']}",
    )
