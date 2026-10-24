import json
from pathlib import Path
import shutil
import subprocess

import gymnasium as gym
import pytest

from src.visualize import build_game_rules


def test_exported_game_rules_match_all_gymnasium_transitions():
    rules = build_game_rules()
    assert rules["max_steps"] == 100
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    try:
        for state in range(16):
            for action in range(4):
                exported = rules["transitions"][state][action]
                original = env.unwrapped.P[state][action]
                assert len(exported) == len(original)
                for actual, (p, next_state, reward, terminated) in zip(exported, original):
                    assert actual == {"probability": p, "state": next_state,
                                      "reward": reward, "terminated": terminated}
    finally:
        env.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is optional for JS engine tests")
def test_browser_game_rules_outcomes_and_frozen_policy():
    engine = Path(__file__).resolve().parents[1] / "src/free_play.js"
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {FreePlayGame} = require(process.argv[1]);
const rules = JSON.parse(fs.readFileSync(0, 'utf8'));
const map = ['SFFF','FHFH','FFFH','HFFG'];
const table = Array.from({length:16},()=>[0,0,1,0]);
// From state 6, asking right can go down (10), right (hole 7), or up (2).
for (const [draw, expected] of [[.1,10],[.5,7],[.9,2]]) {
    let count=0;
    const game=new FreePlayGame(map,table,rules,6,()=>++count%2 ? 0 : draw);
    const move=game.step();assert.equal(move.action,2);assert.equal(move.to,expected);
}
const before=JSON.stringify(table);
const victory=new FreePlayGame(map,table,rules,14,()=>.5);
victory.step();assert.equal(victory.state,15);assert.equal(victory.outcome,'success');
assert.equal(victory.totalReward,1);assert.equal(victory.step(),null);
assert.equal(victory.transitions.length,1);
const defeat=new FreePlayGame(map,table,rules,4,()=>.5);
defeat.step();assert.equal(defeat.state,5);assert.equal(defeat.outcome,'hole');
assert.equal(JSON.stringify(table),before);
assert.ok(Object.isFrozen(victory.qTable[0]));
const left=Array.from({length:16},()=>[1,0,0,0]);
const timeout=new FreePlayGame(map,left,rules,0,()=>.5);
for(let i=0;i<100;i++)timeout.step();
assert.equal(timeout.state,0);assert.equal(timeout.outcome,'timeout');
assert.equal(timeout.transitions.length,100);assert.equal(timeout.step(),null);
for(const bad of [-1,5,7,11,12,15,16,0.5]) {
    assert.throws(()=>new FreePlayGame(map,table,rules,bad));
}
// Policy is an owned copy, not an alias of the supplied table.
table[14][2]=99;assert.equal(victory.qTable[14][2],1);
console.log('Browser engine: slips, safe starts, outcomes, time limit, immutable Q-table passed');
"""
    result = subprocess.run([shutil.which("node"), "-e", script, str(engine)],
                            input=json.dumps(build_game_rules()), capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr


def test_template_exports_free_play_without_external_script(tmp_path):
    from src.visualize import build_history, save_history

    history = build_history(n_episodes=1, eval_episodes=1, n_replays=1)
    page = save_history(history, tmp_path).read_text(encoding="utf-8")
    assert "__GAME_ENGINE__" not in page
    assert "class FreePlayGame" in page
    assert 'id="free-tab"' in page
    assert '<script src=' not in page
    saved = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert saved["game_rules"] == build_game_rules()


def test_old_history_is_upgraded_without_changing_checkpoints(tmp_path):
    from src.visualize import build_history, save_history

    history = build_history(n_episodes=1, eval_episodes=1, n_replays=1)
    history.pop("game_rules")
    history["schema_version"] = 1
    original = json.dumps(history["checkpoints"])
    save_history(history, tmp_path)
    saved = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert json.dumps(saved["checkpoints"]) == original
    assert "game_rules" not in history
