"use strict";

MCMC.registerAlgorithm("HaRAM", {
  description: "Hamiltonian Repelling-Attracting Metropolis",
  
  about: () => {
    window.open("https://sidvishwanath.com");
  },
  
  init: (self) => {
    self.leapfrogSteps = 20;
    self.dt = 0.5;
    self.gamma = 0.1;
  },
  
  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },
  
  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.05, 0.9).step(0.025).name("Leapfrog &Delta;t");
    folder.add(self, "gamma", 0.00, 0.50).step(0.01).name("&gamma;");
    folder.open();
  },
  
  step: (self, visualizer) => {
    const q0 = self.chain.last();
    const p0 = MultivariateNormal.getSample(self.dim);
    
    // use leapfrog integration to find proposal
    const exp_gamma = 2.71828 ** (self.gamma * self.dt / 2);
    const exp_invgamma = 2.71828 ** (-self.gamma * self.dt / 2);
    // const exp_invgamma = gamma ** -1;
    const q = q0.copy();
    const p = p0.copy();
    const trajectory = [q.copy()];
    
    for (let i = 0; i < self.leapfrogSteps; i++) {
      p.
      increment(p.scale(1 - exp_invgamma)).
      increment(self.gradLogDensity(q).scale(self.dt / 2));
      q.increment(p.scale(self.dt));
      p.
      increment(p.scale(1 - exp_invgamma)).
      increment(self.gradLogDensity(q).scale(self.dt / 2));
      trajectory.push(q.copy());
    }
    
    for (let i = 0; i < self.leapfrogSteps; i++) {
      p.
      increment(p.scale(1 - exp_gamma)).
      increment(self.gradLogDensity(q).scale(self.dt / 2));
      q.increment(p.scale(self.dt));
      p.
      increment(p.scale(1 - exp_gamma)).
      increment(self.gradLogDensity(q).scale(self.dt / 2));
      trajectory.push(q.copy());
    }
    
    // add integrated trajectory to visualizer animation queue
    visualizer.queue.push({
      type: "proposal",
      proposal: q,
      trajectory: trajectory,
      initialMomentum: p0,
    });
    
    // calculate acceptance ratio
    const H0 = -self.logDensity(q0) + p0.norm2() / 2;
    const H = -self.logDensity(q) + p.norm2() / 2;
    const logAcceptRatio = -H + H0;
    
    // accept or reject proposal
    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(q.copy());
      visualizer.queue.push({ type: "accept", proposal: q });
    } else {
      self.chain.push(q0.copy());
      visualizer.queue.push({ type: "reject", proposal: q });
    }
  },
});
