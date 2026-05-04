% MATLAB script: Plot n(d) and dn/dd(d) on the same figure with dual y-axes

clear; close all; clc;

% === 1. Define all parameters (YOU NEED TO SET REALISTIC VALUES HERE) ===
pc1     = 760*5.974;          % example value
pc2     = 760*163.3;          % example
pc3     = 760*0.1744;          % example
c4      = 0.08958;          % example
c5      = 1.509;          % example
c6      = 2.638;          % example
V       = 6000;          % voltage or speed? [units consistent with 0.1 cm]
p       = 760;         % pressure or probability? 
epsilon_r = 5;        % relative permittivity (example: FR4 ~4.5)

d_min   = 0;            % μm, mm, cm — choose consistent unit
d_max   = 0.1;            % example range — adjust based on your parameters
n_points = 4000;

% Precompute constant part
const = 0.1 / epsilon_r;     % cm  (assuming 0.1 cm in the expression)

% === 2. Create d vector ===
d = linspace(d_max * 0.02, d_max, n_points)';   % avoid starting exactly at 0 if problematic

% === 3. Compute s, the common thickness-like term ===
s = const + d;           % s = 0.1 cm / ε_r + d

% === 4. Compute the two main terms ===
term1 = pc1 * exp(-pc2 * s ./ V);

term2_inner_exp = (c4 * p * s ./ V).^c5;
term2_exp_part  = exp(-term2_inner_exp);
term2_sigmoid   = 1 - term2_exp_part;

term2_power     = (V ./ (p * s)).^c6;

term2 = pc3 * term2_sigmoid .* term2_power;

% The expression inside the exp
inside = term1 + term2;

% The function n(d)
n = exp(d .* inside);

% === 5. Compute dn/dd (fully expanded) ===
dAd_d = - (pc1 * pc2 / V) .* exp(-pc2 * s ./ V);           % = d(term1)/dd

% Derivatives of term2 parts
u = term2_inner_exp;               % u = (c4 p s / V)^c5
du_dd = c5 .* (c4 * p ./ V) .* (c4 * p * s ./ V).^(c5-1);

w  = term2_power;                  % w = (V/(p s))^c6
dw_dd = -c6 .* w ./ s;

dterm2_dd = pc3 * (...
    exp(-u) .* du_dd .* w + ...               % sigmoid rising part
    term2_sigmoid .* dw_dd ...                % power-law decaying part
);

% Full derivative inside: d(inside)/dd
dinside_dd = dAd_d + dterm2_dd;

% dn/dd = n * (inside + d * dinside/dd)
dndd = n .* (inside + d .* dinside_dd);

% === 6. Plotting ===
figure('Color','w','Position',[300 180 920 580]);

yyaxis left
plot(d, n, 'b-', 'LineWidth', 2.1);
ylabel('n(d)', 'Color','b', 'FontSize',13);
grid on; hold on;

yyaxis right
plot(d, dndd, 'r--', 'LineWidth', 2.0);
ylabel('dn/dd', 'Color','r', 'FontSize',13);

xlabel('d   (cm)', 'FontSize',13);
title('n(d) and dn/dd(d)   —   dual axis plot', 'FontSize',14);
set(gca, 'FontSize',11);

% Optional: mark where derivative ≈ 0
[~, idx_max] = max(n);
xline(d(idx_max), ':k', 'd at max n', 'LabelOrientation','horizontal', ...
      'LabelHorizontalAlignment','right', 'Alpha',0.6);

legend('n(d)', 'dn/dd', 'Location','best');
hold off;