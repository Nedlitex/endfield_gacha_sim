"""State management, serialization, and initialization."""

import base64
import json
import zlib

import streamlit as st

from banner import Banner, BannerTemplate, EndFieldBannerTemplate, Operator
from gacha import Config, Player
from strategy import deserialize_strategy, is_legacy_strategy
from ui.defaults import (
    create_default_banners,
    create_default_operators,
    create_default_strategies,
)


def serialize_state() -> str:
    """Serialize current state to compressed base64 string."""
    # Serialize run results if present
    run_results_data = None
    if st.session_state.get("run_results"):
        results = st.session_state.run_results
        run_results_data = {
            "player": results["player"].model_dump(),
            "paid_draws": results["paid_draws"],
            "total_draws": results["total_draws"],
            "num_experiments": results["num_experiments"],
            "banners": results["banners"],
            "main_operators": results.get("main_operators", []),
            "total_banner_count": results.get(
                "total_banner_count", len(results["banners"])
            ),
        }

    state = {
        "operators": [op.model_dump() for op in st.session_state.operators],
        "banners": [banner.model_dump() for banner in st.session_state.banners],
        "banner_templates": [
            t.model_dump() for t in st.session_state.get("banner_templates", [])
        ],
        "box": st.session_state.box.model_dump(),
        "config": st.session_state.config.model_dump(),
        "strategies": [s.model_dump() for s in st.session_state.strategies],
        "current_strategy_idx": st.session_state.current_strategy_idx,
        "run_banner_enabled": st.session_state.get("run_banner_enabled", {}),
        "run_banner_strategies": st.session_state.get("run_banner_strategies", {}),
        "auto_banner_count": st.session_state.get("auto_banner_count", 0),
        "auto_banner_template_idx": st.session_state.get("auto_banner_template_idx", 0),
        "auto_banner_strategy_idx": st.session_state.get("auto_banner_strategy_idx", 0),
        "run_results": run_results_data,
    }
    json_str = json.dumps(state, ensure_ascii=False)
    compressed = zlib.compress(json_str.encode(), level=9)
    return base64.urlsafe_b64encode(compressed).decode()


def deserialize_state(encoded: str) -> dict:
    """Deserialize state from compressed base64 string."""
    compressed = base64.urlsafe_b64decode(encoded.encode())
    try:
        # Try decompressing (new format)
        json_str = zlib.decompress(compressed).decode()
    except zlib.error:
        # Fall back to old uncompressed format
        json_str = compressed.decode()
    return json.loads(json_str)


def update_url():
    """Update URL parameters with current state."""
    encoded = serialize_state()
    st.query_params["state"] = encoded


def initialize_session_state():
    """Initialize session state from URL or defaults."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        params = st.query_params
        if "state" in params:
            try:
                state = deserialize_state(params["state"])
                st.session_state.operators = [
                    Operator(**op) for op in state.get("operators", [])
                ]
                st.session_state.banners = [
                    Banner(**banner) for banner in state.get("banners", [])
                ]
                st.session_state.box = Player(**state.get("box", {}))
                # Load config
                if "config" in state:
                    st.session_state.config = Config(**state.get("config", {}))
                else:
                    st.session_state.config = Config()
                # Load strategies (skip legacy format strategies)
                if "strategies" in state:
                    new_strategies = []
                    for s in state.get("strategies", []):
                        if not is_legacy_strategy(s):
                            new_strategies.append(deserialize_strategy(s))
                    # If all strategies were legacy, use defaults
                    if not new_strategies:
                        new_strategies = create_default_strategies()
                    st.session_state.strategies = new_strategies
                    st.session_state.current_strategy_idx = min(
                        state.get("current_strategy_idx", 0),
                        len(new_strategies) - 1,
                    )
                else:
                    st.session_state.strategies = create_default_strategies()
                    st.session_state.current_strategy_idx = 0
                # Load banner templates
                if "banner_templates" in state and state["banner_templates"]:
                    st.session_state.banner_templates = [
                        BannerTemplate(**t) for t in state["banner_templates"]
                    ]
                else:
                    st.session_state.banner_templates = [
                        EndFieldBannerTemplate.model_copy(deep=True)
                    ]
                # Load run state
                st.session_state.run_banner_enabled = state.get(
                    "run_banner_enabled", {}
                )
                st.session_state.run_banner_strategies = state.get(
                    "run_banner_strategies", {}
                )
                # Load auto banner config
                st.session_state.auto_banner_count = state.get("auto_banner_count", 0)
                st.session_state.auto_banner_template_idx = state.get(
                    "auto_banner_template_idx", 0
                )
                st.session_state.auto_banner_strategy_idx = state.get(
                    "auto_banner_strategy_idx", 0
                )
                # Load run results
                if "run_results" in state and state["run_results"]:
                    run_data = state["run_results"]
                    st.session_state.run_results = {
                        "player": Player(**run_data["player"]),
                        "paid_draws": run_data["paid_draws"],
                        "total_draws": run_data["total_draws"],
                        "num_experiments": run_data["num_experiments"],
                        "banners": run_data["banners"],
                        "main_operators": run_data.get("main_operators", []),
                        "total_banner_count": run_data.get(
                            "total_banner_count", len(run_data["banners"])
                        ),
                    }
                else:
                    st.session_state.run_results = None
            except Exception:
                _initialize_defaults()
        else:
            _initialize_defaults()


def _initialize_defaults():
    """Initialize session state with default values."""
    st.session_state.operators = create_default_operators()
    st.session_state.banners = create_default_banners()
    st.session_state.banner_templates = [EndFieldBannerTemplate.model_copy(deep=True)]
    st.session_state.box = Player()
    st.session_state.config = Config()
    st.session_state.strategies = create_default_strategies()
    st.session_state.current_strategy_idx = 0
    st.session_state.run_banner_enabled = {}
    st.session_state.run_banner_strategies = {}
    st.session_state.auto_banner_count = 0
    st.session_state.auto_banner_template_idx = 0
    st.session_state.auto_banner_strategy_idx = 0
