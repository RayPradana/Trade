"""Tests for bot.ml_models – Machine Learning & AI trading models."""

import unittest
from typing import Dict

from bot.analysis import Candle
from bot.ml_models import (
    AnomalyDetection,
    DeepLearningPrediction,
    EngineeredFeatures,
    GradientBoostPrediction,
    MarketPrediction,
    MarketSentiment,
    NeuralNetPrediction,
    NewsSentiment,
    ParameterTuning,
    PatternRecognitionAI,
    RandomForestPrediction,
    RegimeDetection,
    RLAction,
    SocialSentiment,
    StrategyOptimization,
    analyze_market_sentiment,
    analyze_news_sentiment,
    analyze_social_sentiment,
    auto_tune_parameters,
    deep_learning_predict,
    detect_anomalies,
    detect_regime,
    engineer_features,
    gradient_boost_predict,
    neural_net_predict,
    optimize_strategy,
    predict_market,
    random_forest_predict,
    recognize_patterns_ai,
    rl_trading_signal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(closes, start_ts=1000, volume=100.0):
    """Build candles from a list of close prices."""
    candles = []
    for i, c in enumerate(closes):
        candles.append(Candle(
            timestamp=start_ts + i * 60,
            open=c * 0.999,
            high=c * 1.005,
            low=c * 0.995,
            close=c,
            volume=volume,
        ))
    return candles


def _uptrend_candles(n=30, start=100.0, step=1.0, volume=100.0):
    """Generate an uptrend price series."""
    return _make_candles([start + i * step for i in range(n)], volume=volume)


def _downtrend_candles(n=30, start=100.0, step=1.0, volume=100.0):
    """Generate a downtrend price series."""
    return _make_candles([start - i * step for i in range(n)], volume=volume)


def _flat_candles(n=30, price=100.0, volume=100.0):
    """Generate a flat/sideways price series."""
    return _make_candles([price] * n, volume=volume)


def _volatile_candles(n=30, base=100.0, amplitude=10.0, volume=100.0):
    """Generate a volatile/oscillating price series."""
    closes = [base + amplitude * (1 if i % 2 == 0 else -1) for i in range(n)]
    return _make_candles(closes, volume=volume)


# ═══════════════════════════════════════════════════════════════════════════
#  1. Reinforcement Learning Trading Models
# ═══════════════════════════════════════════════════════════════════════════

class RLTradingTests(unittest.TestCase):
    """Tests for rl_trading_signal."""

    def test_insufficient_data(self):
        result = rl_trading_signal(_make_candles([100, 101]))
        self.assertEqual(result.action, "hold")
        self.assertIn("insufficient", result.reason)

    def test_uptrend_signal(self):
        result = rl_trading_signal(_uptrend_candles(20))
        self.assertIsInstance(result, RLAction)
        self.assertIn(result.action, ("buy", "sell", "hold"))
        self.assertIn("buy", result.q_values)
        self.assertIn("sell", result.q_values)
        self.assertIn("hold", result.q_values)

    def test_downtrend_signal(self):
        result = rl_trading_signal(_downtrend_candles(20, start=200.0))
        self.assertIn(result.action, ("buy", "sell", "hold"))

    def test_exploration_rate_stored(self):
        result = rl_trading_signal(_uptrend_candles(20), exploration_rate=0.25)
        self.assertEqual(result.exploration_rate, 0.25)

    def test_q_values_populated(self):
        result = rl_trading_signal(_uptrend_candles(20))
        self.assertEqual(set(result.q_values.keys()), {"buy", "sell", "hold"})


# ═══════════════════════════════════════════════════════════════════════════
#  2. Neural Network Prediction Models
# ═══════════════════════════════════════════════════════════════════════════

class NeuralNetTests(unittest.TestCase):
    """Tests for neural_net_predict."""

    def test_insufficient_data(self):
        result = neural_net_predict(_make_candles([100] * 5))
        self.assertEqual(result.predicted_direction, "neutral")

    def test_prediction_output(self):
        result = neural_net_predict(_uptrend_candles(25))
        self.assertIsInstance(result, NeuralNetPrediction)
        self.assertIn(result.predicted_direction, ("up", "down"))
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_layer_outputs_populated(self):
        result = neural_net_predict(_uptrend_candles(25), hidden_size=4)
        self.assertEqual(len(result.layer_outputs), 4)

    def test_predicted_return_range(self):
        result = neural_net_predict(_downtrend_candles(25, start=200))
        self.assertGreaterEqual(result.predicted_return, -0.05)
        self.assertLessEqual(result.predicted_return, 0.05)


# ═══════════════════════════════════════════════════════════════════════════
#  3. Gradient Boosting Models
# ═══════════════════════════════════════════════════════════════════════════

class GradientBoostTests(unittest.TestCase):
    """Tests for gradient_boost_predict."""

    def test_insufficient_data(self):
        result = gradient_boost_predict(_make_candles([100] * 5))
        self.assertEqual(result.action, "hold")

    def test_prediction_output(self):
        result = gradient_boost_predict(_uptrend_candles(20))
        self.assertIsInstance(result, GradientBoostPrediction)
        self.assertIn(result.action, ("buy", "sell", "hold"))
        self.assertIsInstance(result.feature_importances, dict)

    def test_trees_agree_pct(self):
        result = gradient_boost_predict(_uptrend_candles(20))
        self.assertGreaterEqual(result.trees_agree_pct, 0.0)
        self.assertLessEqual(result.trees_agree_pct, 1.0)

    def test_feature_importances(self):
        result = gradient_boost_predict(_uptrend_candles(20))
        self.assertIn("trend", result.feature_importances)
        self.assertIn("volatility", result.feature_importances)


# ═══════════════════════════════════════════════════════════════════════════
#  4. Random Forest Models
# ═══════════════════════════════════════════════════════════════════════════

class RandomForestTests(unittest.TestCase):
    """Tests for random_forest_predict."""

    def test_insufficient_data(self):
        result = random_forest_predict(_make_candles([100] * 5))
        self.assertEqual(result.action, "hold")

    def test_prediction_output(self):
        result = random_forest_predict(_uptrend_candles(20))
        self.assertIsInstance(result, RandomForestPrediction)
        self.assertIn(result.action, ("buy", "sell", "hold"))
        self.assertEqual(result.n_trees, 20)

    def test_class_probabilities_sum(self):
        result = random_forest_predict(_uptrend_candles(20))
        total = sum(result.class_probabilities.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_custom_tree_count(self):
        result = random_forest_predict(_uptrend_candles(20), n_trees=50)
        self.assertEqual(result.n_trees, 50)


# ═══════════════════════════════════════════════════════════════════════════
#  5. Deep Learning Models
# ═══════════════════════════════════════════════════════════════════════════

class DeepLearningTests(unittest.TestCase):
    """Tests for deep_learning_predict."""

    def test_insufficient_data(self):
        result = deep_learning_predict(_make_candles([100] * 5))
        self.assertEqual(result.predicted_direction, "neutral")

    def test_prediction_output(self):
        result = deep_learning_predict(_uptrend_candles(25))
        self.assertIsInstance(result, DeepLearningPrediction)
        self.assertIn(result.predicted_direction, ("up", "down"))

    def test_layer_activations(self):
        result = deep_learning_predict(_uptrend_candles(25), n_layers=3)
        self.assertEqual(len(result.layer_activations), 3)

    def test_attention_weights(self):
        result = deep_learning_predict(_uptrend_candles(25), lookback=20)
        self.assertEqual(len(result.attention_weights), 20)
        # Attention weights should sum to ~1 (softmax)
        total = sum(result.attention_weights)
        self.assertAlmostEqual(total, 1.0, places=2)


# ═══════════════════════════════════════════════════════════════════════════
#  6. Feature Engineering Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class FeatureEngineeringTests(unittest.TestCase):
    """Tests for engineer_features."""

    def test_insufficient_data(self):
        result = engineer_features(_make_candles([100] * 5))
        self.assertEqual(result.feature_count, 0)

    def test_features_generated(self):
        result = engineer_features(_uptrend_candles(25))
        self.assertIsInstance(result, EngineeredFeatures)
        self.assertGreater(result.feature_count, 0)
        self.assertIn("return_mean", result.features)
        self.assertIn("momentum_5", result.features)

    def test_feature_selection(self):
        result = engineer_features(_uptrend_candles(25), importance_threshold=0.01)
        self.assertIsInstance(result.selected_features, list)

    def test_flat_data(self):
        result = engineer_features(_flat_candles(25))
        self.assertIsInstance(result, EngineeredFeatures)
        self.assertGreater(result.feature_count, 0)


# ═══════════════════════════════════════════════════════════════════════════
#  7. Market Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════════════

class MarketSentimentTests(unittest.TestCase):
    """Tests for analyze_market_sentiment."""

    def test_insufficient_data(self):
        result = analyze_market_sentiment(_make_candles([100] * 5))
        self.assertEqual(result.sentiment, "neutral")

    def test_uptrend_sentiment(self):
        result = analyze_market_sentiment(_uptrend_candles(30, step=2.0))
        self.assertIsInstance(result, MarketSentiment)
        self.assertIn(result.sentiment, ("greed", "neutral", "fear"))

    def test_components_present(self):
        result = analyze_market_sentiment(_uptrend_candles(30))
        self.assertIn("trend", result.components)
        self.assertIn("volatility", result.components)
        self.assertIn("momentum", result.components)

    def test_score_range(self):
        result = analyze_market_sentiment(_downtrend_candles(30, start=200))
        self.assertGreaterEqual(result.score, -1.0)
        self.assertLessEqual(result.score, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  8. News Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════════════

class NewsSentimentTests(unittest.TestCase):
    """Tests for analyze_news_sentiment."""

    def test_no_headlines(self):
        result = analyze_news_sentiment([])
        self.assertEqual(result.sentiment, "neutral")

    def test_positive_headlines(self):
        headlines = [
            "Bitcoin rally continues with bullish surge",
            "Crypto adoption gains momentum, profits soar",
        ]
        result = analyze_news_sentiment(headlines)
        self.assertIsInstance(result, NewsSentiment)
        self.assertEqual(result.sentiment, "positive")
        self.assertGreater(result.polarity, 0)

    def test_negative_headlines(self):
        headlines = [
            "Crypto crash leads to massive selloff",
            "Fear of regulation causes decline and dump",
        ]
        result = analyze_news_sentiment(headlines)
        self.assertEqual(result.sentiment, "negative")
        self.assertLess(result.polarity, 0)

    def test_keywords_detected(self):
        headlines = ["bullish rally expected"]
        result = analyze_news_sentiment(headlines)
        self.assertIn("bullish", result.keywords_detected)
        self.assertIn("rally", result.keywords_detected)

    def test_neutral_headlines(self):
        headlines = ["The weather is nice today"]
        result = analyze_news_sentiment(headlines)
        self.assertEqual(result.sentiment, "neutral")


# ═══════════════════════════════════════════════════════════════════════════
#  9. Social Media Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════════════

class SocialSentimentTests(unittest.TestCase):
    """Tests for analyze_social_sentiment."""

    def test_insufficient_posts(self):
        result = analyze_social_sentiment([{"text": "bullish"}])
        self.assertEqual(result.sentiment, "neutral")
        self.assertIn("insufficient", result.reason)

    def test_bullish_posts(self):
        posts = [
            {"text": "bullish rally moon", "engagement": 100},
            {"text": "buy buy buy breakout", "engagement": 200},
            {"text": "surge growth adoption", "engagement": 150},
            {"text": "profit gain long", "engagement": 80},
            {"text": "bullish bullish moon", "engagement": 120},
        ]
        result = analyze_social_sentiment(posts)
        self.assertIsInstance(result, SocialSentiment)
        self.assertEqual(result.sentiment, "bullish")
        self.assertGreater(result.bullish_ratio, result.bearish_ratio)

    def test_bearish_posts(self):
        posts = [
            {"text": "crash dump sell", "engagement": 100},
            {"text": "bearish decline fear", "engagement": 200},
            {"text": "scam fraud hack", "engagement": 150},
            {"text": "short selloff loss", "engagement": 80},
            {"text": "bearish bearish crash", "engagement": 120},
        ]
        result = analyze_social_sentiment(posts)
        self.assertEqual(result.sentiment, "bearish")

    def test_volume_tracked(self):
        posts = [{"text": "bullish", "engagement": 10}] * 10
        result = analyze_social_sentiment(posts)
        self.assertEqual(result.volume, 10)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Regime Detection Using ML
# ═══════════════════════════════════════════════════════════════════════════

class RegimeDetectionTests(unittest.TestCase):
    """Tests for detect_regime."""

    def test_insufficient_data(self):
        result = detect_regime(_make_candles([100] * 5))
        self.assertEqual(result.regime, "unknown")

    def test_trending_up(self):
        result = detect_regime(_uptrend_candles(35, step=2.0))
        self.assertIsInstance(result, RegimeDetection)
        self.assertIn(result.regime, ("trending_up", "trending_down", "ranging", "volatile", "quiet"))

    def test_volatile_regime(self):
        result = detect_regime(_volatile_candles(35, amplitude=15.0))
        self.assertIn(result.regime, ("volatile", "ranging", "trending_up", "trending_down", "quiet"))

    def test_features_populated(self):
        result = detect_regime(_uptrend_candles(35))
        self.assertIn("trend", result.features_used)
        self.assertIn("volatility", result.features_used)

    def test_confidence_range(self):
        result = detect_regime(_uptrend_candles(35))
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Adaptive Strategy Optimization
# ═══════════════════════════════════════════════════════════════════════════

class StrategyOptimizationTests(unittest.TestCase):
    """Tests for optimize_strategy."""

    def test_insufficient_data(self):
        result = optimize_strategy(_make_candles([100] * 5))
        self.assertEqual(result.recommended_strategy, "none")

    def test_optimization_output(self):
        result = optimize_strategy(_uptrend_candles(25))
        self.assertIsInstance(result, StrategyOptimization)
        self.assertIn(result.recommended_strategy,
                      ("trend_following", "mean_reversion", "momentum", "scalping", "none"))
        self.assertGreater(len(result.tested_strategies), 0)

    def test_custom_strategies(self):
        result = optimize_strategy(
            _uptrend_candles(25),
            strategies=["trend_following", "momentum"],
        )
        strat_names = [s["strategy"] for s in result.tested_strategies]
        self.assertIn("trend_following", strat_names)
        self.assertIn("momentum", strat_names)

    def test_params_populated(self):
        result = optimize_strategy(_uptrend_candles(25))
        self.assertIsInstance(result.recommended_params, dict)


# ═══════════════════════════════════════════════════════════════════════════
# 12. Auto Parameter Tuning
# ═══════════════════════════════════════════════════════════════════════════

class ParameterTuningTests(unittest.TestCase):
    """Tests for auto_tune_parameters."""

    def test_insufficient_data(self):
        result = auto_tune_parameters(_make_candles([100] * 5))
        self.assertEqual(result.iterations, 0)

    def test_tuning_output(self):
        result = auto_tune_parameters(_uptrend_candles(25))
        self.assertIsInstance(result, ParameterTuning)
        self.assertGreater(result.iterations, 0)
        self.assertIsInstance(result.best_params, dict)

    def test_custom_param_space(self):
        space = {"lookback": (5.0, 30.0), "multiplier": (1.0, 3.0)}
        result = auto_tune_parameters(_uptrend_candles(25), param_space=space)
        self.assertEqual(set(result.search_space.keys()), {"lookback", "multiplier"})

    def test_iterations_count(self):
        result = auto_tune_parameters(_uptrend_candles(25), n_iterations=20)
        self.assertEqual(result.iterations, 20)


# ═══════════════════════════════════════════════════════════════════════════
# 13. Anomaly Detection AI
# ═══════════════════════════════════════════════════════════════════════════

class AnomalyDetectionTests(unittest.TestCase):
    """Tests for detect_anomalies."""

    def test_insufficient_data(self):
        result = detect_anomalies(_make_candles([100] * 5))
        self.assertFalse(result.is_anomalous)

    def test_normal_data(self):
        result = detect_anomalies(_flat_candles(35))
        self.assertIsInstance(result, AnomalyDetection)

    def test_anomalous_spike(self):
        """A sudden spike should be detected as anomalous."""
        prices = [100.0] * 29 + [200.0]  # sudden double
        candles = _make_candles(prices)
        result = detect_anomalies(candles)
        self.assertTrue(result.is_anomalous)
        self.assertGreater(result.anomaly_score, 0.0)
        self.assertGreater(len(result.anomalies), 0)

    def test_anomaly_types(self):
        prices = [100.0] * 29 + [200.0]
        candles = _make_candles(prices)
        result = detect_anomalies(candles)
        types = [a["type"] for a in result.anomalies]
        self.assertTrue(any(t in ("return", "volume", "range") for t in types))


# ═══════════════════════════════════════════════════════════════════════════
# 14. Market Prediction Models
# ═══════════════════════════════════════════════════════════════════════════

class MarketPredictionTests(unittest.TestCase):
    """Tests for predict_market."""

    def test_insufficient_data(self):
        result = predict_market(_make_candles([100] * 5))
        self.assertEqual(result.predicted_direction, "neutral")

    def test_uptrend_prediction(self):
        result = predict_market(_uptrend_candles(25))
        self.assertIsInstance(result, MarketPrediction)
        self.assertIn(result.predicted_direction, ("up", "down", "neutral"))
        self.assertGreater(result.predicted_price, 0)

    def test_prediction_range(self):
        result = predict_market(_uptrend_candles(25))
        low, high = result.prediction_range
        self.assertLessEqual(low, high)

    def test_model_agreement(self):
        result = predict_market(_uptrend_candles(25))
        self.assertGreaterEqual(result.model_agreement, 0.0)
        self.assertLessEqual(result.model_agreement, 1.0)

    def test_models_used(self):
        result = predict_market(_uptrend_candles(25), models=["linear", "momentum"])
        self.assertIn("linear", result.models_used)
        self.assertIn("momentum", result.models_used)


# ═══════════════════════════════════════════════════════════════════════════
# 15. Pattern Recognition AI
# ═══════════════════════════════════════════════════════════════════════════

class PatternRecognitionAITests(unittest.TestCase):
    """Tests for recognize_patterns_ai."""

    def test_insufficient_data(self):
        result = recognize_patterns_ai(_make_candles([100, 101]))
        self.assertEqual(result.action, "hold")

    def test_basic_pattern_detection(self):
        result = recognize_patterns_ai(_uptrend_candles(10))
        self.assertIsInstance(result, PatternRecognitionAI)
        self.assertIn(result.action, ("buy", "sell", "hold"))

    def test_hammer_detection(self):
        """Create a hammer candle (long lower shadow, small body)."""
        candles = [
            Candle(timestamp=1000 + i * 60, open=100.0, high=101.0, low=99.0, close=100.5, volume=100)
            for i in range(4)
        ]
        # Last candle = hammer: long lower shadow
        candles.append(Candle(
            timestamp=1240, open=100.2, high=100.5,
            low=95.0, close=100.3, volume=200,
        ))
        result = recognize_patterns_ai(candles)
        pattern_names = [p["name"] for p in result.patterns]
        self.assertIn("hammer", pattern_names)

    def test_bullish_engulfing(self):
        """Create a bullish engulfing pattern."""
        candles = [
            Candle(timestamp=1000 + i * 60, open=100.0, high=101.0, low=99.0, close=100.0, volume=100)
            for i in range(3)
        ]
        # Bearish candle followed by larger bullish candle
        candles.append(Candle(timestamp=1180, open=100.0, high=100.5, low=98.5, close=99.0, volume=100))
        candles.append(Candle(timestamp=1240, open=98.5, high=101.5, low=98.0, close=101.0, volume=200))
        result = recognize_patterns_ai(candles)
        pattern_names = [p["name"] for p in result.patterns]
        self.assertIn("bullish_engulfing", pattern_names)

    def test_three_white_soldiers(self):
        """Three consecutive bullish candles with higher closes."""
        candles = [
            Candle(timestamp=1000, open=100.0, high=101.0, low=99.0, close=100.0, volume=100),
            Candle(timestamp=1060, open=100.0, high=101.0, low=99.0, close=100.0, volume=100),
            Candle(timestamp=1120, open=100.0, high=101.5, low=99.5, close=101.0, volume=100),
            Candle(timestamp=1180, open=101.0, high=102.5, low=100.5, close=102.0, volume=100),
            Candle(timestamp=1240, open=102.0, high=103.5, low=101.5, close=103.0, volume=100),
        ]
        result = recognize_patterns_ai(candles)
        pattern_names = [p["name"] for p in result.patterns]
        self.assertIn("three_white_soldiers", pattern_names)


if __name__ == "__main__":
    unittest.main()
