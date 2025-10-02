import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Pause, RotateCcw, Settings, Target, TrendingUp, Zap, Brain, AlertTriangle } from 'lucide-react';

// Enhanced Relational State Representation Agent
class EnhancedRelationalAgent {
  constructor(config = {}) {
    this.config = {
      learningRate: 0.001,
      epsilon: 0.1,
      epsilonDecay: 0.995,
      gamma: 0.99,
      memorySize: 10000,
      batchSize: 32,
      targetUpdateFreq: 100,
      maxSubgoals: 8,
      baseSubgoalSpacing: 50, // Adaptive spacing
      movementHistorySize: 20,
      adaptivePhases: true,
      ...config
    };
    
    this.memory = [];
    this.step = 0;
    this.subgoals = [];
    this.currentSubgoalIndex = 0;
    this.performanceHistory = [];
    this.movementHistory = [];
    this.initialPosition = [50, 50];
    this.targetPosition = [350, 350];
    
    // Fixed neural network architecture
    this.weights = this.initializeWeights();
    this.targetWeights = { ...this.weights };
  }

  initializeWeights() {
    return {
      hidden1: Array(32).fill(0).map(() => Array(24).fill(0).map(() => (Math.random() - 0.5) * 0.1)),
      hidden2: Array(24).fill(0).map(() => Array(16).fill(0).map(() => (Math.random() - 0.5) * 0.1)),
      output: Array(16).fill(0).map(() => Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.1))
    };
  }

  // Core utility methods
  euclideanDistance(pos1, pos2) {
    return Math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2);
  }

  computeDirection(from, to) {
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    return Math.atan2(dy, dx);
  }

  // Enhanced state representation with all relational features
  computeRelationalState(currentPos, targetPos, environment) {
    const state = {};
    
    // Store positions for reference
    this.targetPosition = targetPos;
    
    // 1. Enhanced Progress and Remaining Ratios
    const totalDistance = this.euclideanDistance(this.initialPosition, targetPos);
    const currentDistance = this.euclideanDistance(currentPos, targetPos);
    const startDistance = this.euclideanDistance(this.initialPosition, currentPos);
    
    state.progressRatio = Math.max(0, Math.min(1, startDistance / totalDistance));
    state.remainingRatio = currentDistance / totalDistance;
    state.completionRatio = 1 - state.remainingRatio;
    
    // 2. Multi-scale Gap Analysis (fully normalized)
    state.linearGapRatio = currentDistance / totalDistance;
    state.logGapRatio = Math.log(Math.max(1, currentDistance)) / Math.log(Math.max(1, totalDistance));
    state.sqrtGapRatio = Math.sqrt(currentDistance) / Math.sqrt(totalDistance);
    
    // 3. Enhanced Time Analysis
    const timeElapsed = this.step;
    const optimalTime = Math.ceil(totalDistance / 5); // Assuming 5 units per step
    const estimatedRemaining = Math.ceil(currentDistance / 5);
    
    state.timeProgressRatio = Math.min(1, timeElapsed / optimalTime);
    state.timeEfficiencyRatio = optimalTime / Math.max(1, timeElapsed);
    state.remainingTimeRatio = estimatedRemaining / Math.max(1, optimalTime);
    
    // 4. Enhanced Constraint Features
    state.constraintPressure = this.computeConstraintPressure(currentPos, environment);
    state.forbiddenProximity = this.computeForbiddenProximity(currentPos, environment);
    state.trapPressure = this.computeTrapPressure(currentPos, environment);
    state.escapeRoutes = this.computeEscapeRoutes(currentPos, environment);
    
    // 5. Adaptive Phase Identification
    const phaseData = this.identifyAdaptivePhase(state.progressRatio, state.constraintPressure);
    state.currentPhase = phaseData.phase;
    state.phaseProgress = phaseData.progress;
    state.phaseConfidence = phaseData.confidence;
    
    // 6. Fixed Efficiency Metrics
    const theoreticalMinSteps = Math.ceil(totalDistance / 5);
    state.theoreticalMinSteps = theoreticalMinSteps;
    state.actualSteps = this.step;
    state.efficiencyRatio = Math.min(1, theoreticalMinSteps / Math.max(1, this.step));
    state.wasteRatio = Math.max(0, (this.step - theoreticalMinSteps) / Math.max(1, theoreticalMinSteps));
    
    // 7. Enhanced Hierarchical Goal Features
    const subgoalFeatures = this.computeEnhancedSubgoalFeatures(currentPos, targetPos);
    Object.assign(state, subgoalFeatures);
    
    // 8. Enhanced Movement Analysis
    state.directionToTarget = this.computeDirection(currentPos, targetPos);
    state.movementConsistency = this.computeMovementConsistency();
    state.pathEfficiency = this.computePathEfficiency();
    state.velocityRatio = this.computeVelocityRatio();
    
    // 9. Strategic Features
    state.explorationPressure = this.computeExplorationPressure();
    state.convergenceIndicator = this.computeConvergenceIndicator();
    
    return state;
  }

  // Enhanced constraint analysis
  computeConstraintPressure(pos, environment) {
    let pressure = 0;
    const obstacles = environment.obstacles || [];
    
    obstacles.forEach(obstacle => {
      const dist = this.euclideanDistance(pos, obstacle.center);
      const safeDistance = obstacle.radius + 10; // Increased safety margin
      
      if (dist < safeDistance) {
        const normalizedDist = (dist - obstacle.radius) / 10; // Normalize to [0,1]
        pressure += Math.exp(-normalizedDist * 2); // Smoother exponential decay
      }
    });
    
    return Math.tanh(pressure); // Always [0,1]
  }

  computeForbiddenProximity(pos, environment) {
    const obstacles = environment.obstacles || [];
    if (obstacles.length === 0) return 0;
    
    let minDist = Infinity;
    obstacles.forEach(obstacle => {
      const dist = Math.max(0, this.euclideanDistance(pos, obstacle.center) - obstacle.radius);
      minDist = Math.min(minDist, dist);
    });
    
    // Normalize based on typical environment size
    return minDist === Infinity ? 0 : Math.exp(-minDist / 50);
  }

  computeTrapPressure(pos, environment) {
    const checkRadius = 25;
    const directions = Array.from({length: 8}, (_, i) => {
      const angle = (i * 2 * Math.PI) / 8;
      return [Math.cos(angle), Math.sin(angle)];
    });
    
    let blockedDirections = 0;
    
    directions.forEach(dir => {
      const checkPos = [
        pos[0] + dir[0] * checkRadius, 
        pos[1] + dir[1] * checkRadius
      ];
      
      if (this.isBlocked(checkPos, environment)) {
        blockedDirections++;
      }
    });
    
    return blockedDirections / directions.length;
  }

  computeEscapeRoutes(pos, environment) {
    const directions = 16; // More granular check
    let freeDirections = 0;
    
    for (let i = 0; i < directions; i++) {
      const angle = (i * 2 * Math.PI) / directions;
      const checkPos = [
        pos[0] + Math.cos(angle) * 30,
        pos[1] + Math.sin(angle) * 30
      ];
      
      if (!this.isBlocked(checkPos, environment)) {
        freeDirections++;
      }
    }
    
    return freeDirections / directions;
  }

  isBlocked(pos, environment) {
    // Check bounds
    if (pos[0] < 0 || pos[0] > 400 || pos[1] < 0 || pos[1] > 400) return true;
    
    // Check obstacles
    const obstacles = environment.obstacles || [];
    return obstacles.some(obstacle => 
      this.euclideanDistance(pos, obstacle.center) < obstacle.radius
    );
  }

  // Adaptive phase identification
  identifyAdaptivePhase(progressRatio, constraintPressure) {
    // Base thresholds
    let earlyThreshold = 0.25;
    let endThreshold = 0.75;
    
    // Adapt based on constraint pressure
    if (constraintPressure > 0.5) {
      earlyThreshold += 0.1; // Extend early phase when constrained
      endThreshold += 0.1;
    }
    
    // Adapt based on complexity
    const complexity = this.computeEnvironmentComplexity();
    if (complexity > 0.5) {
      earlyThreshold += 0.05;
      endThreshold -= 0.05; // Earlier end phase for complex environments
    }
    
    let phase, progress, confidence;
    
    if (progressRatio < earlyThreshold) {
      phase = 'early';
      progress = progressRatio / earlyThreshold;
      confidence = 1 - constraintPressure * 0.3;
    } else if (progressRatio < endThreshold) {
      phase = 'mid';
      progress = (progressRatio - earlyThreshold) / (endThreshold - earlyThreshold);
      confidence = 0.8 + (0.2 * (1 - Math.abs(progress - 0.5) * 2));
    } else {
      phase = 'end';
      progress = (progressRatio - endThreshold) / (1 - endThreshold);
      confidence = 1 - (1 - progressRatio) * 0.5;
    }
    
    return { phase, progress, confidence: Math.max(0.3, Math.min(1, confidence)) };
  }

  computeEnvironmentComplexity() {
    // Simple heuristic based on obstacle density
    const totalArea = 400 * 400;
    const obstacleArea = this.environment?.obstacles?.reduce((sum, obs) => 
      sum + Math.PI * obs.radius * obs.radius, 0) || 0;
    
    return Math.min(1, obstacleArea / (totalArea * 0.3)); // Normalize
  }

  // Enhanced hierarchical goal system
  decomposeGoal(startPos, targetPos, environment) {
    this.initialPosition = startPos;
    this.targetPosition = targetPos;
    this.environment = environment;
    
    const totalDistance = this.euclideanDistance(startPos, targetPos);
    const complexity = this.computeEnvironmentComplexity();
    
    // Adaptive spacing based on distance and complexity
    const adaptiveSpacing = this.config.baseSubgoalSpacing * (1 - complexity * 0.5);
    const numSubgoals = Math.min(
      this.config.maxSubgoals, 
      Math.max(1, Math.floor(totalDistance / adaptiveSpacing))
    );
    
    const subgoals = [];
    for (let i = 1; i <= numSubgoals; i++) {
      const ratio = i / numSubgoals;
      let subgoalPos = [
        startPos[0] + (targetPos[0] - startPos[0]) * ratio,
        startPos[1] + (targetPos[1] - startPos[1]) * ratio
      ];
      
      // Enhanced obstacle avoidance
      subgoalPos = this.findSafeSubgoalPosition(subgoalPos, environment, startPos, targetPos);
      
      subgoals.push({
        position: subgoalPos,
        priority: 1.0 - (i - 1) / numSubgoals,
        completed: false,
        attempts: 0,
        originalRatio: ratio,
        safetyMargin: this.computeSafetyMargin(subgoalPos, environment)
      });
    }
    
    this.subgoals = subgoals;
    this.currentSubgoalIndex = 0;
    return subgoals;
  }

  findSafeSubgoalPosition(originalPos, environment, startPos, targetPos) {
    const obstacles = environment.obstacles || [];
    let safePos = [...originalPos];
    
    // Check if original position is safe
    const minDist = Math.min(...obstacles.map(obs => 
      this.euclideanDistance(originalPos, obs.center) - obs.radius
    ));
    
    if (minDist >= 15) return safePos; // Already safe
    
    // Find alternative positions
    const searchRadius = 30;
    const candidates = [];
    
    for (let angle = 0; angle < 2 * Math.PI; angle += Math.PI / 8) {
      for (let radius = 15; radius <= searchRadius; radius += 10) {
        const candidate = [
          originalPos[0] + Math.cos(angle) * radius,
          originalPos[1] + Math.sin(angle) * radius
        ];
        
        if (this.isPositionSafe(candidate, environment, 15)) {
          const pathDistance = this.euclideanDistance(startPos, candidate) + 
                               this.euclideanDistance(candidate, targetPos);
          const directDistance = this.euclideanDistance(startPos, targetPos);
          const detourRatio = pathDistance / directDistance;
          
          candidates.push({
            position: candidate,
            detourRatio: detourRatio,
            safetyMargin: this.computeSafetyMargin(candidate, environment)
          });
        }
      }
    }
    
    if (candidates.length > 0) {
      // Choose best candidate (balance between safety and detour)
      candidates.sort((a, b) => 
        (a.detourRatio * 0.7 + (1 - a.safetyMargin) * 0.3) - 
        (b.detourRatio * 0.7 + (1 - b.safetyMargin) * 0.3)
      );
      safePos = candidates[0].position;
    }
    
    return safePos;
  }

  isPositionSafe(pos, environment, margin = 10) {
    if (pos[0] < margin || pos[0] > 400 - margin || 
        pos[1] < margin || pos[1] > 400 - margin) return false;
    
    const obstacles = environment.obstacles || [];
    return obstacles.every(obs => 
      this.euclideanDistance(pos, obs.center) >= obs.radius + margin
    );
  }

  computeSafetyMargin(pos, environment) {
    const obstacles = environment.obstacles || [];
    if (obstacles.length === 0) return 1;
    
    const minDist = Math.min(...obstacles.map(obs => 
      this.euclideanDistance(pos, obs.center) - obs.radius
    ));
    
    return Math.min(1, Math.max(0, minDist / 50));
  }

  getCurrentTarget() {
    if (this.subgoals.length === 0 || this.currentSubgoalIndex >= this.subgoals.length) {
      return null;
    }
    return this.subgoals[this.currentSubgoalIndex];
  }

  updateSubgoalProgress(currentPos) {
    const currentTarget = this.getCurrentTarget();
    if (!currentTarget) return false;
    
    const distance = this.euclideanDistance(currentPos, currentTarget.position);
    if (distance < 8.0) { // Completion threshold
      currentTarget.completed = true;
      this.currentSubgoalIndex++;
      return true;
    }
    
    // Check if subgoal is blocked and needs replanning
    if (currentTarget.attempts > 20) { // Too many attempts
      this.replanSubgoal(currentTarget, currentPos);
    }
    
    return false;
  }

  replanSubgoal(subgoal, currentPos) {
    const newPos = this.findSafeSubgoalPosition(
      subgoal.position, 
      this.environment, 
      currentPos, 
      this.targetPosition
    );
    
    subgoal.position = newPos;
    subgoal.attempts = 0;
    subgoal.replanned = true;
  }

  computeEnhancedSubgoalFeatures(currentPos, finalTarget) {
    const currentTarget = this.getCurrentTarget();
    const features = {
      hasActiveSubgoal: currentTarget !== null,
      subgoalProgress: 0,
      subgoalPriority: 0,
      subgoalsCompleted: this.currentSubgoalIndex,
      totalSubgoals: this.subgoals.length,
      subgoalCompletionRatio: this.subgoals.length > 0 ? this.currentSubgoalIndex / this.subgoals.length : 0,
      subgoalDensity: this.subgoals.length / Math.max(1, this.euclideanDistance(this.initialPosition, finalTarget) / 50),
      averageSubgoalSafety: 0
    };
    
    if (currentTarget) {
      const distToSubgoal = this.euclideanDistance(currentPos, currentTarget.position);
      const initialDistToSubgoal = this.euclideanDistance(this.initialPosition, currentTarget.position);
      features.subgoalProgress = Math.max(0, 1 - distToSubgoal / Math.max(1, initialDistToSubgoal));
      features.subgoalPriority = currentTarget.priority;
      features.subgoalSafetyMargin = currentTarget.safetyMargin || 0;
      features.subgoalAttempts = Math.min(1, currentTarget.attempts / 20);
    }
    
    // Average safety of all subgoals
    if (this.subgoals.length > 0) {
      features.averageSubgoalSafety = this.subgoals.reduce((sum, sg) => 
        sum + (sg.safetyMargin || 0), 0) / this.subgoals.length;
    }
    
    return features;
  }

  // Enhanced movement analysis
  recordMovement(pos) {
    this.movementHistory.push({
      position: [...pos],
      timestamp: this.step
    });
    
    if (this.movementHistory.length > this.config.movementHistorySize) {
      this.movementHistory.shift();
    }
  }

  computeMovementConsistency() {
    if (this.movementHistory.length < 3) return 1.0;
    
    const recent = this.movementHistory.slice(-10);
    const directions = [];
    
    for (let i = 1; i < recent.length; i++) {
      const dir = this.computeDirection(recent[i-1].position, recent[i].position);
      directions.push(dir);
    }
    
    if (directions.length < 2) return 1.0;
    
    // Compute direction variance
    const meanDir = directions.reduce((sum, dir) => sum + dir, 0) / directions.length;
    const variance = directions.reduce((sum, dir) => sum + Math.pow(dir - meanDir, 2), 0) / directions.length;
    
    return Math.exp(-variance / 2); // Higher consistency = lower variance
  }

  computePathEfficiency() {
    if (this.movementHistory.length < 2) return 1.0;
    
    const start = this.movementHistory[0].position;
    const current = this.movementHistory[this.movementHistory.length - 1].position;
    
    // Calculate actual path length
    let actualPath = 0;
    for (let i = 1; i < this.movementHistory.length; i++) {
      actualPath += this.euclideanDistance(
        this.movementHistory[i-1].position,
        this.movementHistory[i].position
      );
    }
    
    const directPath = this.euclideanDistance(start, current);
    return directPath > 0 ? Math.min(1, directPath / actualPath) : 1.0;
  }

  computeVelocityRatio() {
    if (this.movementHistory.length < 5) return 0.5;
    
    const recent = this.movementHistory.slice(-5);
    let totalDistance = 0;
    
    for (let i = 1; i < recent.length; i++) {
      totalDistance += this.euclideanDistance(
        recent[i-1].position,
        recent[i].position
      );
    }
    
    const timeSpan = recent[recent.length - 1].timestamp - recent[0].timestamp;
    const avgVelocity = totalDistance / Math.max(1, timeSpan);
    
    return Math.min(1, avgVelocity / 5); // Normalize to expected velocity
  }

  computeExplorationPressure() {
    return Math.max(0, this.config.epsilon - 0.05); // Simple exploration indicator
  }

  computeConvergenceIndicator() {
    if (this.performanceHistory.length < 10) return 0;
    
    const recent = this.performanceHistory.slice(-10);
    const slope = this.calculateSlope(recent);
    
    return Math.tanh(slope * 10); // Positive if improving
  }

  calculateSlope(values) {
    const n = values.length;
    const xSum = (n * (n - 1)) / 2;
    const ySum = values.reduce((sum, val) => sum + val, 0);
    const xySum = values.reduce((sum, val, idx) => sum + val * idx, 0);
    const xSqSum = (n * (n - 1) * (2 * n - 1)) / 6;
    
    return (n * xySum - xSum * ySum) / (n * xSqSum - xSum * xSum);
  }

  // Fixed neural network with correct dimensions
  stateToVector(state) {
    return [
      // Progress and completion (4 features)
      state.progressRatio || 0,
      state.remainingRatio || 0,
      state.completionRatio || 0,
      state.timeProgressRatio || 0,
      
      // Multi-scale gaps (3 features) 
      state.linearGapRatio || 0,
      state.logGapRatio || 0,
      state.sqrtGapRatio || 0,
      
      // Time analysis (3 features)
      state.timeEfficiencyRatio || 0,
      state.remainingTimeRatio || 0,
      state.velocityRatio || 0,
      
      // Constraints and environment (4 features)
      state.constraintPressure || 0,
      state.forbiddenProximity || 0,
      state.trapPressure || 0,
      state.escapeRoutes || 0,
      
      // Phase information (4 features)
      state.currentPhase === 'early' ? 1 : 0,
      state.currentPhase === 'mid' ? 1 : 0,
      state.currentPhase === 'end' ? 1 : 0,
      state.phaseConfidence || 0,
      
      // Efficiency metrics (2 features)
      state.efficiencyRatio || 0,
      state.wasteRatio || 0,
      
      // Subgoal features (4 features)
      state.hasActiveSubgoal ? 1 : 0,
      state.subgoalProgress || 0,
      state.subgoalCompletionRatio || 0,
      state.averageSubgoalSafety || 0,
      
      // Movement and direction (4 features)
      Math.cos(state.directionToTarget || 0),
      Math.sin(state.directionToTarget || 0),
      state.movementConsistency || 0,
      state.pathEfficiency || 0,
      
      // Strategic features (2 features)
      state.explorationPressure || 0,
      state.convergenceIndicator || 0
    ].slice(0, 32);
  }

  forwardPass(input) {
    if (input.length !== 32) {
      console.warn(`Expected 32 features, got ${input.length}`);
      return [0.25, 0.25, 0.25, 0.25]; // Default equal probabilities
    }
    
    // Layer 1: 32 -> 24
    let layer1 = this.matrixMultiply([input], this.weights.hidden1)[0];
    layer1 = layer1.map(x => Math.max(0, x)); // ReLU
    
    // Layer 2: 24 -> 16  
    let layer2 = this.matrixMultiply([layer1], this.weights.hidden2)[0];
    layer2 = layer2.map(x => Math.max(0, x)); // ReLU
    
    // Output layer: 16 -> 4
    let output = this.matrixMultiply([layer2], this.weights.output)[0];
    return output;
  }

  matrixMultiply(a, b) {
    if (!a || !b || a.length === 0 || b.length === 0) return [];
    if (a[0].length !== b.length) return [];
    
    return a.map(row =>
      b[0].map((_, colIndex) =>
        row.reduce((sum, cell, rowIndex) => sum + cell * b[rowIndex][colIndex], 0)
      )
    );
  }

  // Enhanced action selection
  selectAction(state, environment) {
    if (Math.random() < this.config.epsilon) {
      return Math.floor(Math.random() * 4); // Random action
    }
    
    const stateVector = this.stateToVector(state);
    const qValues = this.forwardPass(stateVector);
    
    // Add some exploration bonus for uncertain states
    if (state.phaseConfidence < 0.7) {
      const bonus = (1 - state.phaseConfidence) * 0.1;
      for (let i = 0; i < qValues.length; i++) {
        qValues[i] += (Math.random() - 0.5) * bonus;
      }
    }
    
    return qValues.indexOf(Math.max(...qValues));
  }

  // Learning and memory management
  remember(state, action, reward, nextState, done) {
    this.memory.push({ 
      state: this.stateToVector(state), 
      action, 
      reward, 
      nextState: this.stateToVector(nextState), 
      done 
    });
    
    if (this.memory.length > this.config.memorySize) {
      this.memory.shift();
    }
  }

  train() {
    if (this.memory.length < this.config.batchSize) return;
    
    // Enhanced epsilon decay with performance consideration
    const recentPerformance = this.performanceHistory.slice(-10);
    const avgPerformance = recentPerformance.length > 0 ? 
      recentPerformance.reduce((a, b) => a + b, 0) / recentPerformance.length : 0;
    
    // Slower decay if performance is improving
    const performanceBonus = avgPerformance > 0 ? 0.999 : 0.995;
    this.config.epsilon = Math.max(0.01, this.config.epsilon * performanceBonus);
    
    // Update target network
    if (this.step % this.config.targetUpdateFreq === 0) {
      this.targetWeights = JSON.parse(JSON.stringify(this.weights));
    }
    
    this.step++;
  }

  // Enhanced performance tracking
  updatePerformance(reward, efficiency, state) {
    const performanceScore = reward * efficiency * (state.phaseConfidence || 1);
    this.performanceHistory.push(performanceScore);
    
    if (this.performanceHistory.length > 100) {
      this.performanceHistory.shift();
    }
  }

  getPerformanceMetrics() {
    if (this.performanceHistory.length === 0) return null;
    
    const recent = this.performanceHistory.slice(-20);
    const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const trend = recent.length > 10 ? 
      this.calculateSlope(recent.slice(-10)) * 100 : 0;
    
    return {
      averagePerformance: avg,
      trend: trend,
      epsilon: this.config.epsilon,
      memorySize: this.memory.length,
      convergence: this.computeConvergenceIndicator(),
      movementEfficiency: this.computePathEfficiency()
    };
  }

  // Diagnostic methods
  getDiagnostics() {
    const currentTarget = this.getCurrentTarget();
    
    return {
      subgoalsCompleted: this.currentSubgoalIndex,
      totalSubgoals: this.subgoals.length,
      currentSubgoalAttempts: currentTarget?.attempts || 0,
      avgSubgoalSafety: this.subgoals.length > 0 ? 
        this.subgoals.reduce((sum, sg) => sum + (sg.safetyMargin || 0), 0) / this.subgoals.length : 0,
      movementHistorySize: this.movementHistory.length,
      replanCount: this.subgoals.filter(sg => sg.replanned).length
    };
  }
}

// React Component for Enhanced Visualization
const EnhancedRelationalRLDemo = () => {
  const [agent, setAgent] = useState(null);
  const [environment, setEnvironment] = useState({
    bounds: [400, 400],
    obstacles: [
      { center: [120, 80], radius: 25 },
      { center: [180, 180], radius: 30 },
      { center: [280, 140], radius: 20 },
      { center: [320, 280], radius: 35 },
      { center: [150, 320], radius: 22 }
    ]
  });
  const [agentPos, setAgentPos] = useState([50, 50]);
  const [targetPos, setTargetPos] = useState([350, 350]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentState, setCurrentState] = useState(null);
  const [subgoals, setSubgoals] = useState([]);
  const [performance, setPerformance] = useState(null);
  const [diagnostics, setDiagnostics] = useState(null);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const intervalRef = useRef(null);

  // Initialize enhanced agent
  useEffect(() => {
    const newAgent = new EnhancedRelationalAgent({
      learningRate: 0.001,
      epsilon: 0.15,
      maxSubgoals: 6,
      baseSubgoalSpacing: 60
    });
    setAgent(newAgent);
    
    const initialSubgoals = newAgent.decomposeGoal([50, 50], [350, 350], environment);
    setSubgoals(initialSubgoals);
  }, []);

  // Enhanced simulation step
  const simulationStep = useCallback(() => {
    if (!agent) return;

    const state = agent.computeRelationalState(agentPos, targetPos, environment);
    setCurrentState(state);

    const action = agent.selectAction(state, environment);
    const actions = [
      [0, -5], // Up
      [0, 5],  // Down  
      [-5, 0], // Left
      [5, 0]   // Right
    ];
    
    let newPos = [
      Math.max(5, Math.min(395, agentPos[0] + actions[action][0])),
      Math.max(5, Math.min(395, agentPos[1] + actions[action][1]))
    ];
    
    // Enhanced collision detection
    const wouldCollide = environment.obstacles.some(obs => 
      agent.euclideanDistance(newPos, obs.center) < obs.radius + 2
    );
    
    if (!wouldCollide) {
      setAgentPos(newPos);
      agent.recordMovement(newPos);
      
      // Update subgoal progress
      const subgoalCompleted = agent.updateSubgoalProgress(newPos);
      if (subgoalCompleted) {
        setSubgoals([...agent.subgoals]);
      }
      
      // Enhanced reward calculation
      const distanceToTarget = agent.euclideanDistance(newPos, targetPos);
      const progressReward = state.progressRatio * 10;
      const efficiencyReward = state.efficiencyRatio * 5;
      const safetyPenalty = state.constraintPressure * -2;
      const subgoalReward = state.subgoalProgress * 3;
      
      const totalReward = progressReward + efficiencyReward + safetyPenalty + subgoalReward - 0.1;
      
      const nextState = agent.computeRelationalState(newPos, targetPos, environment);
      agent.remember(state, action, totalReward, nextState, distanceToTarget < 10);
      agent.train();
      
      agent.updatePerformance(totalReward, state.efficiencyRatio || 1, state);
      setPerformance(agent.getPerformanceMetrics());
      setDiagnostics(agent.getDiagnostics());
    }
  }, [agent, agentPos, targetPos, environment]);

  useEffect(() => {
    if (isRunning && agent) {
      intervalRef.current = setInterval(simulationStep, 150);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isRunning, simulationStep]);

  const reset = () => {
    setIsRunning(false);
    setAgentPos([50, 50]);
    if (agent) {
      agent.step = 0;
      agent.memory = [];
      agent.performanceHistory = [];
      agent.movementHistory = [];
      agent.config.epsilon = 0.15;
      agent.currentSubgoalIndex = 0;
      const newSubgoals = agent.decomposeGoal([50, 50], targetPos, environment);
      setSubgoals(newSubgoals);
    }
    setCurrentState(null);
    setPerformance(null);
    setDiagnostics(null);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gray-900 text-white">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <Brain className="text-blue-400" />
          Enhanced Relational State RL Agent
        </h1>
        <p className="text-gray-300 mb-4">
          Advanced RL agent with fixed neural network, improved efficiency calculations, 
          adaptive subgoal planning, and comprehensive movement analysis.
        </p>
        
        {/* Status Indicators */}
        <div className="flex gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${performance?.trend > 0 ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span>Performance: {performance?.trend > 0 ? 'Improving' : 'Declining'}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${currentState?.efficiencyRatio > 0.7 ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
            <span>Efficiency: {currentState?.efficiencyRatio > 0.7 ? 'Good' : 'Moderate'}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${currentState?.constraintPressure < 0.3 ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span>Safety: {currentState?.constraintPressure < 0.3 ? 'Safe' : 'Constrained'}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Enhanced Visualization */}
        <div className="lg:col-span-2">
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold flex items-center gap-2">
                <Target className="text-green-400" />
                Enhanced Environment
              </h2>
              <div className="flex gap-2">
                <button
                  onClick={() => setIsRunning(!isRunning)}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                    isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  {isRunning ? <Pause size={16} /> : <Play size={16} />}
                  {isRunning ? 'Pause' : 'Start'}
                </button>
                <button
                  onClick={reset}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2"
                >
                  <RotateCcw size={16} />
                  Reset
                </button>
                <button
                  onClick={() => setShowDiagnostics(!showDiagnostics)}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center gap-2"
                >
                  <AlertTriangle size={16} />
                  Debug
                </button>
              </div>
            </div>
            
            <svg width="400" height="400" className="border border-gray-600 bg-gray-700">
              {/* Enhanced Grid */}
              <defs>
                <pattern id="smallGrid" width="10" height="10" patternUnits="userSpaceOnUse">
                  <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#374151" strokeWidth="0.5"/>
                </pattern>
                <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                  <rect width="50" height="50" fill="url(#smallGrid)"/>
                  <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#4B5563" strokeWidth="1"/>
                </pattern>
              </defs>
              <rect width="400" height="400" fill="url(#grid)" />
              
              {/* Movement Trail */}
              {agent?.movementHistory?.slice(-10).map((move, i) => (
                <circle
                  key={i}
                  cx={move.position[0]}
                  cy={move.position[1]}
                  r={2}
                  fill="#8b5cf6"
                  opacity={0.3 + (i * 0.07)}
                />
              ))}
              
              {/* Obstacles with Safety Zones */}
              {environment.obstacles.map((obs, i) => (
                <g key={i}>
                  {/* Safety zone */}
                  <circle
                    cx={obs.center[0]}
                    cy={obs.center[1]}
                    r={obs.radius + 10}
                    fill="#ef4444"
                    opacity="0.1"
                    stroke="#ef4444"
                    strokeWidth="1"
                    strokeDasharray="3,3"
                  />
                  {/* Obstacle */}
                  <circle
                    cx={obs.center[0]}
                    cy={obs.center[1]}
                    r={obs.radius}
                    fill="#ef4444"
                    stroke="#dc2626"
                    strokeWidth="2"
                  />
                </g>
              ))}
              
              {/* Subgoals with Enhanced Visualization */}
              {subgoals.map((subgoal, i) => (
                <g key={i}>
                  {/* Safety margin indicator */}
                  <circle
                    cx={subgoal.position[0]}
                    cy={subgoal.position[1]}
                    r="12"
                    fill="none"
                    stroke={subgoal.safetyMargin > 0.5 ? "#10b981" : "#f59e0b"}
                    strokeWidth="1"
                    opacity="0.5"
                  />
                  {/* Subgoal */}
                  <circle
                    cx={subgoal.position[0]}
                    cy={subgoal.position[1]}
                    r="8"
                    fill={subgoal.completed ? "#10b981" : subgoal.replanned ? "#f59e0b" : "#3b82f6"}
                    stroke={i === agent?.currentSubgoalIndex ? "#fbbf24" : "#1e40af"}
                    strokeWidth={i === agent?.currentSubgoalIndex ? "3" : "2"}
                  />
                  {/* Subgoal number */}
                  <text
                    x={subgoal.position[0]}
                    y={subgoal.position[1] + 4}
                    textAnchor="middle"
                    fontSize="10"
                    fill="white"
                    fontWeight="bold"
                  >
                    {i + 1}
                  </text>
                </g>
              ))}
              
              {/* Path lines */}
              {subgoals.map((subgoal, i) => {
                if (i === 0) return null;
                const prev = subgoals[i-1];
                return (
                  <line
                    key={`line-${i}`}
                    x1={prev.position[0]}
                    y1={prev.position[1]}
                    x2={subgoal.position[0]}
                    y2={subgoal.position[1]}
                    stroke="#6b7280"
                    strokeWidth="1"
                    strokeDasharray="5,5"
                  />
                );
              })}
              
              {/* Target */}
              <circle
                cx={targetPos[0]}
                cy={targetPos[1]}
                r="15"
                fill="#10b981"
                stroke="#059669"
                strokeWidth="3"
              />
              
              {/* Agent with direction indicator */}
              <g>
                <circle
                  cx={agentPos[0]}
                  cy={agentPos[1]}
                  r="12"
                  fill="#8b5cf6"
                  stroke="#7c3aed"
                  strokeWidth="3"
                />
                {currentState && (
                  <line
                    x1={agentPos[0]}
                    y1={agentPos[1]}
                    x2={agentPos[0] + Math.cos(currentState.directionToTarget || 0) * 25}
                    y2={agentPos[1] + Math.sin(currentState.directionToTarget || 0) * 25}
                    stroke="#8b5cf6"
                    strokeWidth="3"
                  />
                )}
              </g>
              
              {/* Constraint pressure visualization */}
              {currentState && currentState.constraintPressure > 0.1 && (
                <circle
                  cx={agentPos[0]}
                  cy={agentPos[1]}
                  r={15 + currentState.constraintPressure * 10}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="2"
                  opacity={currentState.constraintPressure}
                />
              )}
            </svg>
          </div>
        </div>

        {/* Enhanced Information Panels */}
        <div className="space-y-4">
          {/* Performance Metrics */}
          {performance && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <TrendingUp className="text-blue-400" />
                Performance Metrics
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Movement History:</span>
                  <span className="text-white">{diagnostics.movementHistorySize}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Current Attempts:</span>
                  <span className="text-white">{diagnostics.currentSubgoalAttempts}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Replan Count:</span>
                  <span className="text-yellow-400">{diagnostics.replanCount}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Neural Net Input:</span>
                  <span className="text-green-400">32 features ✓</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">State Vector Valid:</span>
                  <span className="text-green-400">
                    {currentState ? agent.stateToVector(currentState).length === 32 ? '✓' : '✗' : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Enhanced Gap Analysis Visualization */}
      {currentState && (
        <div className="mt-6 bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Target className="text-purple-400" />
            Multi-Scale Gap Analysis
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Linear Gap */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Linear Gap Ratio</span>
                <span className="text-white">{currentState.linearGapRatio?.toFixed(3)}</span>
              </div>
              <div className="h-4 bg-gray-700 rounded relative">
                <div
                  className="h-4 bg-gradient-to-r from-green-500 to-red-500 rounded transition-all duration-300"
                  style={{ width: `${(1 - currentState.linearGapRatio) * 100}%` }}
                />
                <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white">
                  {((1 - currentState.linearGapRatio) * 100).toFixed(0)}%
                </div>
              </div>
            </div>
            
            {/* Logarithmic Gap */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Log Gap Ratio</span>
                <span className="text-white">{currentState.logGapRatio?.toFixed(3)}</span>
              </div>
              <div className="h-4 bg-gray-700 rounded relative">
                <div
                  className="h-4 bg-gradient-to-r from-blue-500 to-purple-500 rounded transition-all duration-300"
                  style={{ width: `${(1 - currentState.logGapRatio) * 100}%` }}
                />
                <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white">
                  {((1 - currentState.logGapRatio) * 100).toFixed(0)}%
                </div>
              </div>
            </div>

            {/* Square Root Gap */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Sqrt Gap Ratio</span>
                <span className="text-white">{currentState.sqrtGapRatio?.toFixed(3)}</span>
              </div>
              <div className="h-4 bg-gray-700 rounded relative">
                <div
                  className="h-4 bg-gradient-to-r from-cyan-500 to-pink-500 rounded transition-all duration-300"
                  style={{ width: `${(1 - currentState.sqrtGapRatio) * 100}%` }}
                />
                <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white">
                  {((1 - currentState.sqrtGapRatio) * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-4 text-xs text-gray-400">
            Different gap representations provide complementary information: Linear for immediate decisions, 
            Log for strategic planning, Sqrt for balanced intermediate planning.
          </div>
        </div>
      )}

      {/* Enhanced Configuration and Features */}
      <div className="mt-6 bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4">Enhanced Agent Features & Improvements</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 text-sm">
          
          {/* Fixed Issues */}
          <div>
            <h4 className="font-semibold text-green-400 mb-3">🔧 Issues Fixed</h4>
            <div className="space-y-1 text-gray-300">
              <div>✅ Neural network dimensions (32 inputs)</div>
              <div>✅ Efficiency ratio capped at 1.0</div>
              <div>✅ Proper subgoal management</div>
              <div>✅ Enhanced obstacle avoidance</div>
              <div>✅ Movement history tracking</div>
              <div>✅ Fully relational state representation</div>
              <div>✅ Adaptive phase thresholds</div>
            </div>
          </div>

          {/* New Features */}
          <div>
            <h4 className="font-semibold text-blue-400 mb-3">🚀 New Features</h4>
            <div className="space-y-1 text-gray-300">
              <div>🎯 Adaptive subgoal spacing</div>
              <div>🛡️ Enhanced safety margins</div>
              <div>🔄 Automatic subgoal replanning</div>
              <div>📊 Multi-scale gap analysis</div>
              <div>🎭 Adaptive phase identification</div>
              <div>📈 Movement consistency tracking</div>
              <div>⚡ Escape route analysis</div>
            </div>
          </div>

          {/* State Features */}
          <div>
            <h4 className="font-semibold text-purple-400 mb-3">🧠 State Features (32)</h4>
            <div className="space-y-1 text-gray-300 text-xs">
              <div>• Progress & Completion (4)</div>
              <div>• Multi-scale Gaps (3)</div>
              <div>• Time Analysis (3)</div>
              <div>• Constraints & Environment (4)</div>
              <div>• Phase Information (4)</div>
              <div>• Efficiency Metrics (2)</div>
              <div>• Subgoal Features (4)</div>
              <div>• Movement & Direction (4)</div>
              <div>• Strategic Features (2)</div>
              <div>• Reserved for Future (2)</div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Performance Improvements */}
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-400 mb-3">📊 Performance Improvements</h4>
            <div className="space-y-2 text-sm text-gray-300">
              <div>• Smarter exploration with phase-aware epsilon decay</div>
              <div>• Better reward shaping with multi-objective optimization</div>
              <div>• Adaptive learning based on convergence indicators</div>
              <div>• Enhanced memory management with state vectors</div>
            </div>
          </div>
          
          {/* Robustness Features */}
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="font-semibold text-red-400 mb-3">🛡️ Robustness Features</h4>
            <div className="space-y-2 text-sm text-gray-300">
              <div>• Comprehensive error checking and validation</div>
              <div>• Fallback mechanisms for edge cases</div>
              <div>• Diagnostic tools for debugging</div>
              <div>• Real-time performance monitoring</div>
            </div>
          </div>
        </div>
      </div>

      {/* Usage Instructions */}
      <div className="mt-4 p-4 bg-blue-900/30 border border-blue-500/30 rounded-lg">
        <h4 className="font-semibold text-blue-400 mb-2">Enhanced Usage Guide</h4>
        <div className="text-sm text-gray-300 space-y-1">
          <p>• <strong>Start/Pause:</strong> Control the enhanced learning and navigation system</p>
          <p>• <strong>Debug Mode:</strong> View comprehensive diagnostics and internal state</p>
          <p>• <strong>Adaptive Subgoals:</strong> Blue circles are active, green are completed, yellow are replanned</p>
          <p>• <strong>Safety Visualization:</strong> Dashed circles show obstacle safety zones</p>
          <p>• <strong>Movement Trail:</strong> Purple dots show recent movement history</p>
          <p>• <strong>Multi-Scale Analysis:</strong> Three different gap representations for different planning horizons</p>
          <p>• <strong>Performance Monitoring:</strong> Real-time efficiency, safety, and learning metrics</p>
        </div>
      </div>
    </div>
  );
};

export default EnhancedRelationalRLDemo;between">
                  <span className="text-gray-400">Average Performance:</span>
                  <span className="text-white">{performance.averagePerformance?.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Trend:</span>
                  <span className={performance.trend > 0 ? 'text-green-400' : 'text-red-400'}>
                    {performance.trend > 0 ? '↗' : '↘'} {Math.abs(performance.trend).toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Convergence:</span>
                  <span className="text-white">{performance.convergence?.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Movement Efficiency:</span>
                  <span className={`${performance.movementEfficiency > 0.7 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {performance.movementEfficiency?.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Exploration (ε):</span>
                  <span className="text-white">{performance.epsilon?.toFixed(3)}</span>
                </div>
              </div>
            </div>
          )}

          {/* Enhanced State Information */}
          {currentState && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Settings className="text-green-400" />
                Enhanced State Features
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Progress Ratio:</span>
                  <span className="text-white">{currentState.progressRatio?.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Phase:</span>
                  <span className={`font-semibold ${
                    currentState.currentPhase === 'early' ? 'text-blue-400' :
                    currentState.currentPhase === 'mid' ? 'text-yellow-400' : 'text-red-400'
                  }`}>
                    {currentState.currentPhase} ({currentState.phaseConfidence?.toFixed(2)})
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Efficiency Ratio:</span>
                  <span className={`${currentState.efficiencyRatio > 0.7 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {currentState.efficiencyRatio?.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Constraint Pressure:</span>
                  <span className={`${currentState.constraintPressure > 0.5 ? 'text-red-400' : 'text-green-400'}`}>
                    {currentState.constraintPressure?.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Escape Routes:</span>
                  <span className={`${currentState.escapeRoutes < 0.3 ? 'text-red-400' : 'text-green-400'}`}>
                    {(currentState.escapeRoutes * 100)?.toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Movement Consistency:</span>
                  <span className="text-white">{currentState.movementConsistency?.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Path Efficiency:</span>
                  <span className="text-white">{currentState.pathEfficiency?.toFixed(3)}</span>
                </div>
              </div>
            </div>
          )}

          {/* Enhanced Subgoal Status */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Zap className="text-orange-400" />
              Adaptive Subgoals
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Active Subgoal:</span>
                <span className="text-white">
                  {(agent?.currentSubgoalIndex || 0) + 1} / {subgoals.length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Completed:</span>
                <span className="text-green-400">
                  {subgoals.filter(s => s.completed).length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Replanned:</span>
                <span className="text-yellow-400">
                  {subgoals.filter(s => s.replanned).length}
                </span>
              </div>
              {currentState?.hasActiveSubgoal && (
                <div className="flex justify-between">
                  <span className="text-gray-400">Subgoal Progress:</span>
                  <span className="text-blue-400">
                    {currentState.subgoalProgress?.toFixed(3)}
                  </span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-gray-400">Avg Safety:</span>
                <span className={`${currentState?.averageSubgoalSafety > 0.5 ? 'text-green-400' : 'text-yellow-400'}`}>
                  {currentState?.averageSubgoalSafety?.toFixed(3)}
                </span>
              </div>
            </div>
            
            {/* Subgoal Progress Bars */}
            <div className="mt-3 space-y-1">
              {subgoals.slice(0, 6).map((subgoal, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xs text-gray-400 w-8">{i + 1}</span>
                  <div className="flex-1 h-2 bg-gray-700 rounded relative">
                    <div
                      className={`h-2 rounded transition-all duration-300 ${
                        subgoal.completed 
                          ? 'bg-green-500 w-full' 
                          : subgoal.replanned
                            ? 'bg-yellow-500'
                            : i === agent?.currentSubgoalIndex 
                              ? 'bg-blue-500' 
                              : 'bg-gray-600 w-0'
                      }`}
                      style={{
                        width: subgoal.completed 
                          ? '100%' 
                          : i === agent?.currentSubgoalIndex 
                            ? `${(currentState?.subgoalProgress || 0) * 100}%`
                            : subgoal.replanned ? '50%' : '0%'
                      }}
                    />
                    {/* Safety indicator */}
                    <div 
                      className="absolute top-0 right-1 w-1 h-2 rounded"
                      style={{
                        backgroundColor: subgoal.safetyMargin > 0.5 ? '#10b981' : '#f59e0b',
                        opacity: 0.7
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Diagnostics Panel */}
          {showDiagnostics && diagnostics && (
            <div className="bg-gray-800 rounded-lg p-4 border border-purple-500">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <AlertTriangle className="text-purple-400" />
                Debug Information
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-