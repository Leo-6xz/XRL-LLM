function results = llmLatencyProfile(record, llm, cfg, feeder, opts)
%LLMLATENCYPROFILE  Diagnose WHERE explanation latency is being spent.
%
%   RESULTS = LLMLATENCYPROFILE(RECORD, LLM, CFG, FEEDER, OPTS)
%
%   A single end-to-end latency number cannot tell you whether the model is
%   running on the CPU, whether the prompt is too long, or whether you are
%   paying a model-load cost on every call. This separates them.
%
%   WHAT IS MEASURED
%   ----------------
%     coldCall     first request after idle -- includes model load
%     warmMinimal  trivial prompt, warm model -- pure per-request overhead
%                  plus a handful of tokens. Sets the floor.
%     warmFull     the real explanation prompt, warm model
%     perLevel     summary / standard / detailed separately
%     threeCalls   the legacy path: three independent generations
%     singleCall   all three levels in one generation
%
%   HOW TO READ IT
%   --------------
%   * warmMinimal above ~1 s means per-request overhead dominates and the
%     prompt is not your problem.
%   * warmFull >> warmMinimal means prompt processing dominates. Trim the
%     static grid-context block, which is ~1.5 kB and identical every call.
%   * coldCall >> warmFull on EVERY call means the model is being unloaded
%     between requests. Fix with a longer keep-alive:
%         setx OLLAMA_KEEP_ALIVE 30m     (Windows, then restart Ollama)
%   * If tokens/second is in single digits, the model is on the CPU.
%     Confirm in a shell with `ollama ps` and read the PROCESSOR column.
%     NOTE: the "[yes] GPU available" line in CHECKDEPENDENCIES reports
%     MATLAB's CUDA device, NOT Ollama's. Ollama is a separate process with
%     its own runtime and may be on the CPU while MATLAB sees a GPU.
%
%   REALISTIC EXPECTATIONS
%   ----------------------
%   For llama3:8b generating ~150 tokens:
%       GPU, model resident   ~3-5 s
%       partial GPU offload   ~8-12 s
%       CPU only              ~15-30 s
%   The 2 s target in the project brief is not reachable with a local 8B
%   model on consumer hardware. It IS reachable with a 3B model on a GPU,
%   or with a hosted endpoint.
%
%   See also GENERATELAYEREDEXPLANATIONS, CONNECTLLM.

if nargin < 5 || isempty(opts), opts = struct(); end
if ~isfield(opts,'nTrials'), opts.nTrials = 3; end
if ~isfield(opts,'verbose'), opts.verbose = true; end

if ~llm.available
    error('llmLatencyProfile:noLLM', ...
        'Needs a live LLM backend. connectLLM reported: %s', llm.lastError);
end

results.backend = llm.backend;
results.model   = cfg.llm.model;

% ---------------------------------------------------------------------
% 1. Cold call (may include model load)
% ---------------------------------------------------------------------
t0 = tic;
generate(llm.chat, 'Reply with the single word OK.');
results.coldCallSec = toc(t0);

% ---------------------------------------------------------------------
% 2. Warm minimal -- the floor: per-request overhead + a few tokens
% ---------------------------------------------------------------------
t = zeros(opts.nTrials,1);
for i = 1:opts.nTrials
    t0 = tic;
    generate(llm.chat, 'Reply with the single word OK.');
    t(i) = toc(t0);
end
results.warmMinimalSec = median(t);

% ---------------------------------------------------------------------
% 3. Warm full prompt, per level
% ---------------------------------------------------------------------
levels = {'summary','standard','detailed'};
results.perLevel = struct('level',{},'latencySec',{},'promptChars',{}, ...
                          'outputChars',{},'tokensPerSec',{});
threeCallTotal = 0;

for k = 1:numel(levels)
    L = levels{k};
    p = promptTemplates(record, cfg, feeder, L, 'full');
    t = zeros(opts.nTrials,1);
    out = '';
    for i = 1:opts.nTrials
        t0 = tic;
        out = generate(llm.chat, p);
        t(i) = toc(t0);
    end
    lat = median(t);
    threeCallTotal = threeCallTotal + lat;

    s.level        = L;
    s.latencySec   = lat;
    s.promptChars  = numel(p);
    s.outputChars  = numel(char(out));
    % ~4 chars per token is the usual rough conversion for English.
    s.tokensPerSec = (s.outputChars/4) / max(lat - results.warmMinimalSec, eps);
    results.perLevel(end+1) = s; %#ok<AGROW>
end

results.threeCallSec = threeCallTotal;

% ---------------------------------------------------------------------
% 4. Single-call layered generation
% ---------------------------------------------------------------------
t0 = tic;
try
    [~, m] = generateLayeredExplanations(record, llm, cfg, feeder, ...
        struct('singleCall', true));
    results.singleCallSec = m.totalLatencySec;
    results.singleCallOK  = isfield(m,'singleCall') && m.singleCall;
catch ME
    results.singleCallSec = toc(t0);
    results.singleCallOK  = false;
    results.singleCallErr = ME.message;
end

results.speedup = results.threeCallSec / max(results.singleCallSec, eps);

% ---------------------------------------------------------------------
% Diagnosis
% ---------------------------------------------------------------------
tps = median([results.perLevel.tokensPerSec]);
results.medianTokensPerSec = tps;

if tps < 12
    results.diagnosis = 'CPU inference (or heavy partial offload). Check `ollama ps`.';
elseif results.warmMinimalSec > 1.0
    results.diagnosis = 'Per-request overhead dominates; prompt length is not the issue.';
elseif median([results.perLevel.latencySec]) > 4*results.warmMinimalSec
    results.diagnosis = 'Prompt processing dominates; trim the static grid-context block.';
else
    results.diagnosis = 'Latency is generation-bound and near hardware limits for this model.';
end

if opts.verbose
    fprintf('\n=== LLM latency profile (%s / %s) ===\n', results.backend, results.model);
    fprintf('  Cold call (incl. model load) : %6.2f s\n', results.coldCallSec);
    fprintf('  Warm minimal (floor)         : %6.2f s\n', results.warmMinimalSec);
    fprintf('  %-12s %9s %9s %9s %10s\n','level','latency','prompt','output','tok/s');
    for k = 1:numel(results.perLevel)
        s = results.perLevel(k);
        fprintf('  %-12s %8.2fs %8dch %8dch %10.1f\n', ...
            s.level, s.latencySec, s.promptChars, s.outputChars, s.tokensPerSec);
    end
    fprintf('  Three separate calls         : %6.2f s\n', results.threeCallSec);
    fprintf('  Single-call layered          : %6.2f s  (%.1fx faster)\n', ...
        results.singleCallSec, results.speedup);
    fprintf('\n  Median throughput: %.1f tokens/s\n', tps);
    fprintf('  DIAGNOSIS: %s\n', results.diagnosis);
    fprintf(['\n  Reference for llama3:8b at ~150 output tokens:\n' ...
             '    GPU resident  ~3-5 s   |  partial offload ~8-12 s  |  CPU ~15-30 s\n']);
    fprintf(['  Reminder: checkDependencies reports MATLAB''s GPU, not Ollama''s.\n' ...
             '  Verify with `ollama ps` and read the PROCESSOR column.\n\n']);
end
end
