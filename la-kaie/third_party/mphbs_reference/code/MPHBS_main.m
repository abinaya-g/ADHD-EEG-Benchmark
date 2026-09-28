% MPHBS_MAIN Public MATLAB entry point for the final MPHBS algorithm.
%
% Associated article:
%   MPHBS: A Novel Hybrid Metaheuristic with SARSA-Guided Information
%   Exchange
%
% Authors:
%   Emilio Flores (corresponding author)
%   Escuela de Ingenieria Informatica, Universidad de Valparaiso,
%   Valparaiso, Chile.
%   Facultad de Ingenieria y Negocios, Universidad de las Americas,
%   Santiago, Chile.
%   Email: emilio.flores@postgrado.uv.cl
%
%   Rodrigo Olivares (doctoral supervisor and co-author)
%   Escuela de Ingenieria Informatica, Universidad de Valparaiso,
%   Valparaiso, Chile.
%   Email: rodrigo.olivares@uv.cl
%
% Syntax:
%   [bestValue, bestPosition, convergenceCurve] = MPHBS_main( ...
%       N, T, LB, UB, Dim, F_obj, K, Ni, rho_sub, w_best, gamma, ...
%       fad_before_p2, fad_after_p2)
%
% Inputs:
%   N              Population size.
%   T              Number of outer iterations.
%   LB, UB         Lower/upper bounds, scalar or 1-by-Dim vectors.
%   Dim            Problem dimensionality.
%   F_obj          Objective function handle. It must accept one row vector.
%   K              Number of proxy-screened hybrid candidates per episode.
%   Ni             Number of SARSA-style sub-exploration episodes per subset element.
%   rho_sub        Fraction of ranked candidate memory used by the mediator.
%   w_best         Proxy weight for distance to the current best solution.
%   gamma          SARSA discount factor.
%   fad_before_p2  true/false, applies FAD before the mediator phase.
%   fad_after_p2   true/false, applies FAD after the mediator phase.
%
% Final domain-specific settings used in the manuscript:
%   CEC2022 B-C1: K=7, Ni=3, rho_sub=0.25, w_best=0.050,
%                 gamma=0.30, fad_before_p2=true, fad_after_p2=false.
%   FIR A-C1:     K=3, Ni=9, rho_sub=0.25, w_best=0.050,
%                 gamma=0.30, fad_before_p2=false, fad_after_p2=true.
%
% This file provides the public implementation of the final SARSA-guided
% HBA-MPA architecture used in the revised manuscript experiments.
%
% Equation numbers below refer to the clean revised manuscript.
function [Pg, Best_P, Conv] = MPHBS_main(N, T, LB, UB, Dim, F_obj, K, Ni, rho_sub, w_best, gamma, fad_before_p2, fad_after_p2)

    % =============================================================
    % 0. SAFE BOUNDS AS ROW VECTORS
    % Bound repair follows paper Eqs. (10) and (25).
    % =============================================================
    if isscalar(LB)
        LBv = LB * ones(1, Dim);
    else
        LBv = reshape(LB, 1, []);
    end

    if isscalar(UB)
        UBv = UB * ones(1, Dim);
    else
        UBv = reshape(UB, 1, []);
    end

    % =============================================================
    % 1. INITIALIZATION & PARAMETERS
    % =============================================================

    % --- MPA parameters and population: paper Eqs. (13)-(14) ---
    FADs = 0.2;
    P = 0.5;
    Best_P_MPA = zeros(1, Dim);
    Best_F_MPA = inf;
    X_MPA = initialization(N, Dim, UBv, LBv);
    Ffun_MPA = inf(N, 1);
    X_min_MPA = repmat(LBv, N, 1);
    X_max_MPA = repmat(UBv, N, 1);

    % --- HBA parameters and population: paper Eqs. (1)-(2) ---
    beta = 6;
    C = 2;
    vec_flag = [1, -1];
    Best_P_HBA = zeros(1, Dim);
    Best_F_HBA = inf;
    X_HBA = initialization(N, Dim, UBv, LBv);
    Xnew_HBA = zeros(N, Dim);
    Ffun_HBA = inf(N, 1);
    Ffun_new_HBA = inf(N, 1);

    % --- SARSA and hybrid parameters ---
    Num_Blocks = 5;
    alpha_lr = 0.10;

    % Lightweight local proxy weights; see paper Eqs. (48)-(51).
    w_rank  = 1.00;
    w_novel = 0.30;

    N_sub   = min(N, max(5, round(N * rho_sub)));
    actions = N_sub;

    if Dim < 10
        Num_Blocks = 1;
    end
    Block_Size = ceil(Dim / Num_Blocks);

    % --- Global tracking ---
    Conv = zeros(1, T);
    t = 0;

    % --- Persistent Q-table: paper Eq. (38) ---
    Q = zeros(Dim, actions, 2);

    % =============================================================
    % 2. INITIAL EVALUATION
    % The global best pair is selected according to paper Eq. (33).
    % =============================================================
    X_MPA = apply_bounds(X_MPA, LBv, UBv);
    X_HBA = apply_bounds(X_HBA, LBv, UBv);

    for i = 1:N
        Ffun_MPA(i,1) = F_obj(X_MPA(i,:));
        if Ffun_MPA(i,1) < Best_F_MPA
            Best_F_MPA = Ffun_MPA(i,1);
            Best_P_MPA = X_MPA(i,:);
        end

        Ffun_HBA(i,1) = F_obj(X_HBA(i,:));
        if Ffun_HBA(i,1) < Best_F_HBA
            Best_F_HBA = Ffun_HBA(i,1);
            Best_P_HBA = X_HBA(i,:);
        end
    end

    if Best_F_MPA <= Best_F_HBA
        Pg = Best_F_MPA;
        Best_P = Best_P_MPA;
    else
        Pg = Best_F_HBA;
        Best_P = Best_P_HBA;
    end

    fit_old = Ffun_MPA;
    X_MPA_old = X_MPA;

    % =============================================================
    % 3. MAIN LOOP
    % =============================================================
    while t < T

        % Global progress used by the relaxed acceptance rule.
        prog_global = min(1, t / max(1, T - 1));

        % ---------------------------------------------------------
        % PHASE 1A: heuristic update drivers (once per iteration)
        % HBA: paper Eqs. (3)-(7). MPA: paper Eqs. (15)-(17),
        % (19), (21), and (23).
        % ---------------------------------------------------------
        I = Intensity_HBA_Original(N, Best_P_HBA, X_HBA);
        alpha_hba = C * exp(-(t + 1) / max(1, T));
        Elite = repmat(Best_P_MPA, N, 1);
        CF = (1 - t / max(1, T))^(2 * t / max(1, T));
        RL = 0.05 * levy(N, Dim, 1.5);
        RB = randn(N, Dim);

        % ---------------------------------------------------------
        % PHASE 1B: combined HBA + MPA movement
        % ---------------------------------------------------------
        for i = 1:N
            r = rand();
            F = vec_flag(randi(2));

            for j = 1:Dim
                % HBA movement: paper Eqs. (8)-(9).
                di = Best_P_HBA(j) - X_HBA(i,j);
                if r < 0.5
                    r3 = rand(); r4 = rand(); r5 = rand();
                    Xnew_HBA(i,j) = Best_P_HBA(j) + ...
                                    F * beta * I(i) * Best_P_HBA(j) + ...
                                    F * r3 * alpha_hba * di * ...
                                    abs(cos(2*pi*r4) * (1 - cos(2*pi*r5)));
                else
                    r7 = rand();
                    Xnew_HBA(i,j) = Best_P_HBA(j) + F * r7 * alpha_hba * di;
                end

                % MPA phase-dependent movement: paper Eqs. (18),
                % (20), (22), and (24).
                R = rand();
                if t < T / 3
                    stepsize_ij = RB(i,j) * (Elite(i,j) - RB(i,j) * X_MPA(i,j));
                    X_MPA(i,j) = X_MPA(i,j) + P * R * stepsize_ij;
                elseif t >= T / 3 && t < 2 * T / 3
                    if i > N / 2
                        stepsize_ij = RB(i,j) * (RB(i,j) * Elite(i,j) - X_MPA(i,j));
                        X_MPA(i,j) = Elite(i,j) + P * CF * stepsize_ij;
                    else
                        stepsize_ij = RL(i,j) * (Elite(i,j) - RL(i,j) * X_MPA(i,j));
                        X_MPA(i,j) = X_MPA(i,j) + P * R * stepsize_ij;
                    end
                else
                    stepsize_ij = RL(i,j) * (RL(i,j) * Elite(i,j) - X_MPA(i,j));
                    X_MPA(i,j) = Elite(i,j) + P * CF * stepsize_ij;
                end
            end

            % HBA bound repair and greedy acceptance: paper Eqs. (10)-(11).
            Xnew_HBA(i,:) = apply_bounds(Xnew_HBA(i,:), LBv, UBv);
            Ffun_new_HBA(i,1) = F_obj(Xnew_HBA(i,:));
            if Ffun_new_HBA(i,1) < Ffun_HBA(i,1)
                X_HBA(i,:) = Xnew_HBA(i,:);
                Ffun_HBA(i,1) = Ffun_new_HBA(i,1);
            end
        end

        % ---------------------------------------------------------
        % PHASE 1C: refresh HBA best
        % Paper Eq. (12).
        % ---------------------------------------------------------
        X_HBA = apply_bounds(X_HBA, LBv, UBv);
        [Best_F_HBA, idxH] = min(Ffun_HBA);
        Best_P_HBA = X_HBA(idxH,:);

        % ---------------------------------------------------------
        % PHASE 1D: MPA evaluation + greedy memory update
        % Paper Eqs. (25)-(27).
        % ---------------------------------------------------------
        X_MPA = apply_bounds(X_MPA, LBv, UBv);
        for i = 1:N
            Ffun_MPA(i,1) = F_obj(X_MPA(i,:));
            if Ffun_MPA(i,1) < Best_F_MPA
                Best_F_MPA = Ffun_MPA(i,1);
                Best_P_MPA = X_MPA(i,:);
            end
        end

        Inx = (fit_old < Ffun_MPA);
        Indx = repmat(Inx, 1, Dim);
        X_MPA = Indx .* X_MPA_old + (~Indx) .* X_MPA;
        Ffun_MPA = Inx .* fit_old + (~Inx) .* Ffun_MPA;
        fit_old = Ffun_MPA;
        X_MPA_old = X_MPA;
        % ---------------------------------------------------------
        % PHASE 1E: MPA FADs / Eddy formation with greedy acceptance
        % Paper Eqs. (28)-(30).
        % ---------------------------------------------------------
        if fad_before_p2
            [X_MPA, Ffun_MPA] = apply_mpa_fads_stage(X_MPA, Ffun_MPA, X_min_MPA, X_max_MPA, LBv, UBv, F_obj, FADs, CF);
            [Best_F_MPA, idxM] = min(Ffun_MPA);
            Best_P_MPA = X_MPA(idxM,:);
            fit_old = Ffun_MPA;
            X_MPA_old = X_MPA;
        end

        % Global best before phase 2: paper Eq. (33).
        if Best_F_MPA <= Best_F_HBA
            Pg = Best_F_MPA;
            Best_P = Best_P_MPA;
        else
            Pg = Best_F_HBA;
            Best_P = Best_P_HBA;
        end

        % ---------------------------------------------------------
        % PHASE 2: SARSA-guided dimension-wise information exchange
        % Tensor and Q-table definitions: paper Eqs. (37)-(38).
        % ---------------------------------------------------------
        n_elite = floor(N_sub / 2);
        n_random = N_sub - n_elite;

        [~, sorted_HBA] = sort(Ffun_HBA, 'ascend');
        idx_elite_HBA = sorted_HBA(1:n_elite)';
        remaining_HBA = sorted_HBA(n_elite+1:end);
        if numel(remaining_HBA) < n_random
            idx_rnd_HBA = remaining_HBA(:)';
        else
            idx_rnd_HBA = remaining_HBA(randperm(numel(remaining_HBA), n_random))';
        end
        idx_sub_HBA = [idx_elite_HBA, idx_rnd_HBA];

        [~, sorted_MPA] = sort(Ffun_MPA, 'ascend');
        idx_elite_MPA = sorted_MPA(1:n_elite)';
        remaining_MPA = sorted_MPA(n_elite+1:end);
        if numel(remaining_MPA) < n_random
            idx_rnd_MPA = remaining_MPA(:)';
        else
            idx_rnd_MPA = remaining_MPA(randperm(numel(remaining_MPA), n_random))';
        end
        idx_sub_MPA = [idx_elite_MPA, idx_rnd_MPA];

        % Stabilize action semantics by ranking rows inside each subset.
        FsubH = Ffun_HBA(idx_sub_HBA);
        [~, ordH] = sort(FsubH, 'ascend');
        idx_sub_HBA = idx_sub_HBA(ordH);
        X_HBA_sub = X_HBA(idx_sub_HBA, :);
        F_HBA_sub = Ffun_HBA(idx_sub_HBA);

        FsubM = Ffun_MPA(idx_sub_MPA);
        [~, ordM] = sort(FsubM, 'ascend');
        idx_sub_MPA = idx_sub_MPA(ordM);
        X_MPA_sub = X_MPA(idx_sub_MPA, :);
        F_MPA_sub = Ffun_MPA(idx_sub_MPA);

        result = cat(3, X_HBA_sub, X_MPA_sub);   % [actions x Dim x 2]

        % Project the current global best onto the discrete tensor actions.
        values    = Best_P;
        list_ch   = zeros(1, Dim);
        list_rows = zeros(1, Dim);
        for col = 1:Dim
            [ch0, row0] = get_closest_action(values(col), result, col, actions);
            list_ch(col)   = ch0;
            list_rows(col) = row0;
        end

        f_curr = F_obj(values);
        T_total = N_sub * Ni;

        Shuffled_Dims = randperm(Dim);
        current_dim_ptr = 1;

        for episodes = 1:T_total
            idx_start = current_dim_ptr;
            idx_end   = current_dim_ptr + Block_Size - 1;

            if idx_end > Dim
                part1 = Shuffled_Dims(idx_start:Dim);
                Shuffled_Dims = randperm(Dim);
                n_needed = Block_Size - numel(part1);
                part2 = Shuffled_Dims(1:n_needed);
                active_dims = [part1, part2];
                current_dim_ptr = n_needed + 1;
            else
                active_dims = Shuffled_Dims(idx_start:idx_end);
                current_dim_ptr = idx_end + 1;
                if current_dim_ptr > Dim
                    Shuffled_Dims = randperm(Dim);
                    current_dim_ptr = 1;
                end
            end

            cand_values = cell(K,1);
            cand_ch     = cell(K,1);
            cand_row    = cell(K,1);
            scores      = -inf(K,1);

            % Candidate assembly and action policies: paper Eqs. (39)-(47).
            for k = 1:K
                v_next   = values;
                ch_next  = list_ch;
                row_next = list_rows;

                for col = active_dims
                    switch k
                        case 1
                            [ch_k, row_k] = choose_action_greedy(col, Q, actions);
                        case 2
                            [ch_k, row_k] = choose_action_with_decay(col, Q, actions, episodes, T_total);
                        case 3
                            ch_k  = randi([1 2]);
                            row_k = randi([1 actions]);
                        case 4
                            [ch_k, row_k] = choose_action_secondbest(col, Q, actions);
                        otherwise
                            ep2 = min(T_total, episodes + randi([1, max(1, floor(T_total/10))]));
                            [ch_k, row_k] = choose_action_with_decay(col, Q, actions, ep2, T_total);
                    end

                    v_next(col)   = result(row_k, col, ch_k);
                    ch_next(col)  = ch_k;
                    row_next(col) = row_k;
                end

                % Proxy screening: paper Eqs. (48)-(51).
                mean_rank = mean(row_next(active_dims));
                score_rank = -mean_rank;
                novelty = mean(abs(v_next(active_dims) - values(active_dims)));
                d_best  = norm(v_next(active_dims) - Best_P(active_dims), 2);

                scores(k) = w_rank * score_rank + w_novel * novelty - w_best * d_best;

                cand_values{k} = v_next;
                cand_ch{k}     = ch_next;
                cand_row{k}    = row_next;
            end

            % Proxy winner and selected candidate: paper Eqs. (52)-(53).
            [~, kbest] = max(scores);
            values_next    = cand_values{kbest};
            list_ch_next   = cand_ch{kbest};
            list_rows_next = cand_row{kbest};

            % One expensive evaluation and local reward: paper Eqs. (54)-(55).
            val = F_obj(values_next);
            reward = get_reward(f_curr, val);

            % Relaxed acceptance rule: paper Eqs. (56)-(59).
            prog_local = (episodes - 1) / max(1, T_total - 1);
            if val < f_curr
                accepted = true;
            else
                delta_rel = (val - f_curr) / (abs(f_curr) + eps);
                prob_base = 0.20;
                prob_aceptacion = prob_base * ...
                                  (1 - prog_global)^2 * ...
                                  (0.35 + 0.65 * (1 - prog_local)) * ...
                                  exp(-4 * max(0, delta_rel));
                prob_aceptacion = min(max(prob_aceptacion, 0), 0.25);
                accepted = (rand() < prob_aceptacion);
            end

            % Update Q only on active dimensions: paper Eqs. (60)-(61).
            for col = active_dims
                predict = Q(col, list_rows(col), list_ch(col));

                if accepted
                    next_q = Q(col, list_rows_next(col), list_ch_next(col));
                else
                    next_q = Q(col, list_rows(col), list_ch(col));
                end

                target = reward + gamma * next_q;
                Q(col, list_rows(col), list_ch(col)) = Q(col, list_rows(col), list_ch(col)) + ...
                                                       alpha_lr * (target - predict);
            end

            if accepted
                values    = values_next;
                f_curr    = val;
                list_ch   = list_ch_next;
                list_rows = list_rows_next;

                % Update the ranked subset memories with the accepted candidate.
                if f_curr < max(F_HBA_sub)
                    [~, worst_idx_H] = max(F_HBA_sub);
                    X_HBA_sub(worst_idx_H, :) = values;
                    F_HBA_sub(worst_idx_H) = f_curr;
                end

                if f_curr < max(F_MPA_sub)
                    [~, worst_idx_M] = max(F_MPA_sub);
                    X_MPA_sub(worst_idx_M, :) = values;
                    F_MPA_sub(worst_idx_M) = f_curr;
                end

                result = cat(3, X_HBA_sub, X_MPA_sub);
            end

            % Inject a newly discovered global best into both source populations.
            if val < Pg
                Pg = val;
                Best_P = values_next;

                target_idx_HBA = idx_sub_HBA(randi(numel(idx_sub_HBA)));
                X_HBA(target_idx_HBA, :) = Best_P;
                Ffun_HBA(target_idx_HBA,1) = Pg;

                target_idx_MPA = idx_sub_MPA(randi(numel(idx_sub_MPA)));
                X_MPA(target_idx_MPA, :) = Best_P;
                Ffun_MPA(target_idx_MPA,1) = Pg;
            end
        end

        % Reintegrate the updated subsets into the complete populations.
        X_HBA(idx_sub_HBA, :) = X_HBA_sub;
        Ffun_HBA(idx_sub_HBA) = F_HBA_sub;
        X_MPA(idx_sub_MPA, :) = X_MPA_sub;
        Ffun_MPA(idx_sub_MPA) = F_MPA_sub;

        % Synchronize the global best after reintegration; see paper Eq. (33).
        [Best_F_HBA, idxH] = min(Ffun_HBA);
        Best_P_HBA = X_HBA(idxH,:);
        [Best_F_MPA, idxM] = min(Ffun_MPA);
        Best_P_MPA = X_MPA(idxM,:);

        if Best_F_MPA <= Best_F_HBA
            if Best_F_MPA < Pg
                Pg = Best_F_MPA;
                Best_P = Best_P_MPA;
            end
        else
            if Best_F_HBA < Pg
                Pg = Best_F_HBA;
                Best_P = Best_P_HBA;
            end
        end

        % ---------------------------------------------------------
        % PHASE 3F: optional post-phase2 MPA FADs / Eddy formation
        % Paper Eqs. (28)-(30).
        % ---------------------------------------------------------
        if fad_after_p2
            [X_MPA, Ffun_MPA] = apply_mpa_fads_stage(X_MPA, Ffun_MPA, X_min_MPA, X_max_MPA, LBv, UBv, F_obj, FADs, CF);
            [Best_F_MPA, idxM] = min(Ffun_MPA);
            Best_P_MPA = X_MPA(idxM,:);
            fit_old = Ffun_MPA;
            X_MPA_old = X_MPA;

            if Best_F_MPA <= Best_F_HBA
                if Best_F_MPA < Pg
                    Pg = Best_F_MPA;
                    Best_P = Best_P_MPA;
                end
            else
                if Best_F_HBA < Pg
                    Pg = Best_F_HBA;
                    Best_P = Best_P_HBA;
                end
            end
        end

        t = t + 1;
        Conv(t) = Pg;
    end
end

% =====================================================================
%  HELPERS SARSA
% =====================================================================
function [ch, a] = choose_action_with_decay(pos, Q, actions, current_episode, total_episodes)
    EPS_START = 1.0;
    EPS_END   = 0.15;
    decay_rate = max(1, total_episodes/4);
    eps_val = EPS_END + (EPS_START - EPS_END) * exp(-(current_episode-1)/decay_rate);

    if rand() < eps_val
        ch = randi([1 2]);
        a  = randi([1 actions]);
    else
        v1 = Q(pos,1:actions,1);
        v2 = Q(pos,1:actions,2);
        [m1, a1] = max(v1);
        [m2, a2] = max(v2);
        if m1 >= m2
            ch = 1; a = a1;
        else
            ch = 2; a = a2;
        end
    end
end

function [ch, a] = choose_action_greedy(pos, Q, actions)
    v1 = Q(pos,1:actions,1);
    v2 = Q(pos,1:actions,2);
    [m1, a1] = max(v1);
    [m2, a2] = max(v2);
    if m1 >= m2
        ch = 1; a = a1;
    else
        ch = 2; a = a2;
    end
end

function [ch, a] = choose_action_secondbest(pos, Q, actions)
    v1 = Q(pos,1:actions,1);
    v2 = Q(pos,1:actions,2);
    allv = [v1(:); v2(:)];
    [~, idx] = sort(allv, 'descend');

    if numel(idx) >= 2
        idx2 = idx(2);
    else
        idx2 = idx(1);
    end

    if idx2 <= actions
        ch = 1; a = idx2;
    else
        ch = 2; a = idx2 - actions;
    end
end

function [ch, a] = get_closest_action(target, result, col, actions)
    vals_ch1 = result(:, col, 1);
    vals_ch2 = result(:, col, 2);
    all_vals = [vals_ch1; vals_ch2];

    vmin = min(all_vals);
    vmax = max(all_vals);
    target = min(max(target, vmin), vmax);

    [~, min_idx] = min(abs(all_vals - target));
    if min_idx <= actions
        ch = 1; a = min_idx;
    else
        ch = 2; a = min_idx - actions;
    end
end

% =====================================================================
%  BASICS
% =====================================================================
function X = initialization(N, Dim, UB, LB)
    UB = UB(:)';
    LB = LB(:)';
    R = rand(N, Dim);
    X = repmat(LB, N, 1) + R .* repmat(UB - LB, N, 1);
end

function X = apply_bounds(X, LB, UB)
    if isrow(X)
        X = min(max(X, LB), UB);
    else
        X = min(max(X, repmat(LB, size(X,1), 1)), repmat(UB, size(X,1), 1));
    end
end

function o = levy(n, m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
    sigma_u = (num / den)^(1 / beta);
    u = sigma_u .* randn(n, m);
    v = randn(n, m);
    o = u ./ (abs(v).^(1 / beta));
end

function I = Intensity_HBA_Original(N, Xprey, X)
    di = zeros(N,1);
    S  = zeros(N,1);
    for i = 1:N-1
        di(i) = (norm(X(i,:) - Xprey + eps)).^2;
        S(i)  = (norm(X(i,:) - X(i+1,:) + eps)).^2;
    end
    di(N) = (norm(X(N,:) - Xprey + eps)).^2;
    S(N)  = (norm(X(N,:) - X(1,:) + eps)).^2;
    r2 = rand(N,1);
    I = r2 .* S ./ (4*pi*di + eps);
end


% =====================================================================
%  SHARED MPA FAD STAGE
%  Implements paper Eqs. (28)-(30).
% =====================================================================
function [X_MPA, Ffun_MPA] = apply_mpa_fads_stage(X_MPA, Ffun_MPA, X_min_MPA, X_max_MPA, LBv, UBv, F_obj, FADs, CF)
    if rand() < FADs
        U = rand(size(X_MPA,1), size(X_MPA,2)) < FADs;
        Xcand_MPA = X_MPA + CF * ((X_min_MPA + rand(size(X_MPA,1), size(X_MPA,2)) .* (X_max_MPA - X_min_MPA)) .* U);
    else
        rr = rand();
        Rs = size(X_MPA, 1);
        stepsize_MPA = (FADs * (1 - rr) + rr) * (X_MPA(randperm(Rs), :) - X_MPA(randperm(Rs), :));
        Xcand_MPA = X_MPA + stepsize_MPA;
    end

    Xcand_MPA = apply_bounds(Xcand_MPA, LBv, UBv);
    Fcand_MPA = inf(size(X_MPA,1), 1);
    for ii = 1:size(X_MPA,1)
        Fcand_MPA(ii,1) = F_obj(Xcand_MPA(ii,:));
        if Fcand_MPA(ii,1) < Ffun_MPA(ii,1)
            X_MPA(ii,:) = Xcand_MPA(ii,:);
            Ffun_MPA(ii,1) = Fcand_MPA(ii,1);
        end
    end
end

% =====================================================================
%  REWARD
%  Implements paper Eq. (55).
% =====================================================================
function r = get_reward(f_current, f_new)
    if f_new < f_current
        r = 1;
    elseif f_new == f_current
        r = 0;
    else
        r = -1;
    end
end
