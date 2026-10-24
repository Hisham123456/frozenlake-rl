/* Offline policy execution; transition probabilities are exported by Gymnasium. */
(function (root) {
  'use strict';
  const random = () => {
    if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
      return crypto.getRandomValues(new Uint32Array(1))[0] / 4294967296;
    }
    return Math.random();
  };

  class FreePlayGame {
    constructor(map, qTable, rules, start = 0, rng = random) {
      this.tiles = map.join('');
      if (this.tiles.length !== 16 || !Number.isInteger(start) ||
          !['S', 'F'].includes(this.tiles[start])) {
        throw new Error('Choose a safe starting tile.');
      }
      if (qTable.length !== 16 || qTable.some(row =>
          row.length !== 4 || row.some(value => !Number.isFinite(value)))) {
        throw new Error('Expected a finite 16 by 4 Q-table.');
      }
      // Own a frozen copy: gameplay cannot modify the learned policy.
      this.qTable = Object.freeze(qTable.map(row => Object.freeze([...row])));
      this.rules = rules;
      this.rng = rng;
      this.start = this.state = start;
      this.transitions = [];
      this.totalReward = 0;
      this.done = false;
      this.outcome = null;
    }

    step() {
      if (this.done) return null;
      const values = this.qTable[this.state];
      const best = Math.max(...values);
      const actions = values.map((v, i) => v === best ? i : -1).filter(i => i >= 0);
      // Randomness only breaks ties and samples slips; epsilon remains zero.
      const action = actions[Math.floor(this.rng() * actions.length)];
      const choices = this.rules.transitions[this.state][action];
      const draw = this.rng();
      let cumulative = 0, selected = choices[choices.length - 1];
      for (const choice of choices) {
        cumulative += choice.probability;
        if (draw < cumulative) { selected = choice; break; }
      }
      const transition = {
        from: this.state, to: selected.state, action,
        reward: selected.reward, terminated: selected.terminated,
        truncated: this.transitions.length + 1 >= this.rules.max_steps,
      };
      this.transitions.push(transition);
      this.state = transition.to;
      this.totalReward += transition.reward;
      this.done = transition.terminated || transition.truncated;
      if (this.done) this.outcome = transition.reward > 0 ? 'success' :
        (transition.terminated ? 'hole' : 'timeout');
      return transition;
    }
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = {FreePlayGame};
  else root.FrozenLakePlay = {FreePlayGame};
})(typeof globalThis !== 'undefined' ? globalThis : this);
