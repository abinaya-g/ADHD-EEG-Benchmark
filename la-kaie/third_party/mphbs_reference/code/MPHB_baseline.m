%MPHB_BASELINE HBA-MPA backbone baseline without the mediator.
%
% This function implements the MPHB reference baseline used in the
% manuscript. It keeps the coupled HBA-MPA backbone and MPA FAD mechanism,
% but removes the SARSA-style information-exchange mediator and the matched
% random mirror. It is therefore the backbone-only comparison for evaluating
% the contribution of the mediator.
function [Pg, Best_P, Conv] = MPHB_baseline(N, T, LB, UB, Dim, F_obj)

    % =============================================================
    % 0. SAFE BOUNDS AS ROW VECTORS
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

    % --- MPA Parameters ---
    FADs = 0.2;
    P = 0.5;
    Best_P_MPA = zeros(1, Dim);
    Best_F_MPA = inf;
    X_MPA = initialization(N, Dim, UBv, LBv);
    Ffun_MPA = inf(N, 1);
    X_min_MPA = repmat(LBv, N, 1);
    X_max_MPA = repmat(UBv, N, 1);

    % --- HBA Parameters ---
    beta = 6;
    C = 2;
    vec_flag = [1, -1];
    Best_P_HBA = zeros(1, Dim);
    Best_F_HBA = inf;
    X_HBA = initialization(N, Dim, UBv, LBv);
    Xnew_HBA = zeros(N, Dim);
    Ffun_HBA = inf(N, 1);
    Ffun_new_HBA = inf(N, 1);

    % --- Global tracking ---
    Conv = zeros(1, T);
    t = 0;
    Pg = inf;
    Best_P = zeros(1, Dim);

    % =============================================================
    % 2. INITIAL EVALUATION
    % =============================================================
    X_MPA = apply_bounds(X_MPA, LBv, UBv);
    X_HBA = apply_bounds(X_HBA, LBv, UBv);

    for i = 1:N
        % MPA
        Ffun_MPA(i,1) = F_obj(X_MPA(i,:));
        if Ffun_MPA(i,1) < Best_F_MPA
            Best_F_MPA = Ffun_MPA(i,1);
            Best_P_MPA = X_MPA(i,:);
        end

        % HBA
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

    % MPA marine-memory state.
    fit_old = Ffun_MPA;
    X_MPA_old = X_MPA;

    % =============================================================
    % 3. MAIN LOOP
    % =============================================================
    while t < T

        % ---------------------------------------------------------
        % PHASE 1A: MPA top-predator update.
        % ---------------------------------------------------------
        X_MPA = apply_bounds(X_MPA, LBv, UBv);

        for i = 1:N
            Ffun_MPA(i,1) = F_obj(X_MPA(i,:));
            if Ffun_MPA(i,1) < Best_F_MPA
                Best_F_MPA = Ffun_MPA(i,1);
                Best_P_MPA = X_MPA(i,:);
            end
        end

        % Marine memory #1
        Inx = (fit_old < Ffun_MPA);
        Indx = repmat(Inx, 1, Dim);
        X_MPA    = Indx .* X_MPA_old + (~Indx) .* X_MPA;
        Ffun_MPA = Inx  .* fit_old   + (~Inx)  .* Ffun_MPA;

        fit_old   = Ffun_MPA;
        X_MPA_old = X_MPA;

        % ---------------------------------------------------------
        % Pre-computations for the coupled HBA-MPA movement.
        % ---------------------------------------------------------
        alpha_hba = C * exp(-(t + 1) / max(1, T));
        I_HBA = Intensity_HBA_Original(N, Best_P_HBA, X_HBA);

        Elite = repmat(Best_P_MPA, N, 1);
        CF = (1 - t / max(1, T))^(2 * t / max(1, T));
        RL = 0.05 * levy(N, Dim, 1.5);
        RB = randn(N, Dim);

        % ---------------------------------------------------------
        % PHASE 1B: coupled HBA-MPA movement.
        % ---------------------------------------------------------
        for i = 1:N
            r = rand();
            F = vec_flag(randi(2));

            for j = 1:Dim
                % ---------------- HBA update (original-like) ----------------
                di = Best_P_HBA(j) - X_HBA(i,j);

                if r < 0.5
                    r3 = rand();
                    r4 = rand();
                    r5 = rand();

                    Xnew_HBA(i,j) = Best_P_HBA(j) + ...
                                    F * beta * I_HBA(i) * Best_P_HBA(j) + ...
                                    F * r3 * alpha_hba * di * ...
                                    abs(cos(2*pi*r4) * (1 - cos(2*pi*r5)));
                else
                    r7 = rand();
                    Xnew_HBA(i,j) = Best_P_HBA(j) + F * r7 * alpha_hba * di;
                end

                % ---------------- MPA update (original core) ----------------
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

            % --- HBA greedy acceptance ---
            Xnew_HBA(i,:) = apply_bounds(Xnew_HBA(i,:), LBv, UBv);
            Ffun_new_HBA(i,1) = F_obj(Xnew_HBA(i,:));

            if Ffun_new_HBA(i,1) < Ffun_HBA(i,1)
                Ffun_HBA(i,1) = Ffun_new_HBA(i,1);
                X_HBA(i,:) = Xnew_HBA(i,:);
            end
        end

        % ---------------------------------------------------------
        % PHASE 1C: HBA best-position refresh.
        % ---------------------------------------------------------
        X_HBA = apply_bounds(X_HBA, LBv, UBv);

        [Best_F_HBA, idxH] = min(Ffun_HBA);
        Best_P_HBA = X_HBA(idxH, :);

        % ---------------------------------------------------------
        % PHASE 1D: MPA top-predator update after movement.
        % ---------------------------------------------------------
        X_MPA = apply_bounds(X_MPA, LBv, UBv);

        for i = 1:N
            Ffun_MPA(i,1) = F_obj(X_MPA(i,:));
            if Ffun_MPA(i,1) < Best_F_MPA
                Best_F_MPA = Ffun_MPA(i,1);
                Best_P_MPA = X_MPA(i,:);
            end
        end

        % Marine memory #2
        Inx = (fit_old < Ffun_MPA);
        Indx = repmat(Inx, 1, Dim);
        X_MPA    = Indx .* X_MPA_old + (~Indx) .* X_MPA;
        Ffun_MPA = Inx  .* fit_old   + (~Inx)  .* Ffun_MPA;

        fit_old   = Ffun_MPA;
        X_MPA_old = X_MPA;

        % --- Update global best from heuristics ---
        if Best_F_MPA <= Best_F_HBA
            Pg = Best_F_MPA;
            Best_P = Best_P_MPA;
        else
            Pg = Best_F_HBA;
            Best_P = Best_P_HBA;
        end

        % ---------------------------------------------------------
        % PHASE 4: MPA FADs / eddy-formation update.
        % ---------------------------------------------------------
        fit_old   = Ffun_MPA;
        X_MPA_old = X_MPA;

        CF_end = (1 - t / max(1, T))^(2 * t / max(1, T));

        if rand() < FADs
            U = rand(N, Dim) < FADs;
            X_MPA = X_MPA + CF_end * ...
                ((X_min_MPA + rand(N, Dim) .* (X_max_MPA - X_min_MPA)) .* U);
        else
            rr = rand();
            Rs = size(X_MPA, 1);
            stepsize_MPA = (FADs * (1 - rr) + rr) * ...
                           (X_MPA(randperm(Rs), :) - X_MPA(randperm(Rs), :));
            X_MPA = X_MPA + stepsize_MPA;
        end

        % ---------------------------------------------------------
        % TRACK
        % ---------------------------------------------------------
        t = t + 1;
        Conv(t) = Pg;
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
    if isvector(X) && size(X,1) == 1
        X = min(max(X, LB), UB);
    else
        X = min(max(X, repmat(LB, size(X,1), 1)), repmat(UB, size(X,1), 1));
    end
end

function [o] = levy(n, m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
    sigma_u = (num / den)^(1 / beta);
    u = normrnd(0, sigma_u, n, m);
    v = normrnd(0, 1, n, m);
    z = u ./ (abs(v).^(1 / beta));
    o = z;
end

% =====================================================================
%  HBA ORIGINAL INTENSITY
% =====================================================================
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
