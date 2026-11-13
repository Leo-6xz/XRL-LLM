function llm = connectLLM(cfg)
%CONNECTLLM  Open a connection to the explanation LLM, with a safe fallback.
%
%   LLM = CONNECTLLM(CFG) returns a struct with fields
%     .backend   'ollama' | 'openai' | 'azure' | 'template'
%     .model     model identifier
%     .chat      the underlying chat object ([] for the template backend)
%     .available true if a real LLM is reachable
%
%   Generate text with LLMGENERATE(LLM, PROMPT), never by calling the chat
%   object directly -- that indirection is what lets every downstream stage
%   run unchanged when no model is available.
%
%   CFG.LLM.BACKEND = 'auto' tries Ollama, then OpenAI, then falls back to
%   the deterministic template generator.
%
%   THE TEMPLATE BACKEND IS A FEATURE, NOT A STUB
%   ---------------------------------------------
%   Ablation study 5 in the brief is exactly "template-based generation vs.
%   LLM generation", and the headline claim is a +33.7% quality gain for
%   the LLM. That comparison needs a real template generator to be
%   credible, and it must be a GOOD one -- beating a deliberately poor
%   template proves nothing. GENERATETEMPLATEEXPLANATION is therefore
%   written to be the strongest reasonable non-LLM baseline: it uses the
%   same SHAP attributions, the same topology facts and the same
%   sensitivity numbers, and only lacks the free-form language.
%
%   PREREQUISITES FOR THE OLLAMA PATH
%   ---------------------------------
%     1. Install the MathWorks add-on "Large Language Models (LLMs) with
%        MATLAB" (Home > Add-Ons > Get Add-Ons).
%     2. Install Ollama and start the server, then pull a model:
%            ollama pull llama3:8b
%     3. Verify from a shell:  ollama list
%
%   See also LLMGENERATE, GENERATEEXPLANATION.

if nargin < 1 || isempty(cfg)
    cfg = xrlConfig();
end

backend = lower(cfg.llm.backend);
llm = struct('backend','template','model','none','chat',[], ...
             'available',false,'cfg',cfg,'lastError','');

candidates = {};
switch backend
    case 'auto',     candidates = {'ollama','openai'};
    case 'template', candidates = {};
    otherwise,       candidates = {backend};
end

for c = 1:numel(candidates)
    b = candidates{c};
    try
        switch b
            case 'ollama'
                if isempty(which('ollamaChat'))
                    error('connectLLM:noAddon', ...
                        ['ollamaChat not found. Install the "Large Language ', ...
                         'Models (LLMs) with MATLAB" add-on.']);
                end
                chat = ollamaChat(cfg.llm.model, ...
                    'Endpoint',    cfg.llm.endpoint, ...
                    'TimeOut',     cfg.llm.timeout, ...
                    'Temperature', cfg.llm.temperature);
                % A real round trip is the only reliable liveness test:
                % constructing the object does not contact the server.
                probe = generate(chat, 'Reply with the single word OK.');
                if isempty(probe)
                    error('connectLLM:emptyProbe', 'Empty response from Ollama.');
                end
                llm.backend   = 'ollama';
                llm.model     = cfg.llm.model;
                llm.chat      = chat;
                llm.available = true;
                return

            case 'openai'
                if isempty(which('openAIChat'))
                    error('connectLLM:noAddon', 'openAIChat not found.');
                end
                if isempty(getenv('OPENAI_API_KEY'))
                    error('connectLLM:noKey', 'OPENAI_API_KEY is not set.');
                end
                chat = openAIChat('ModelName', cfg.llm.model, ...
                    'Temperature', cfg.llm.temperature);
                llm.backend   = 'openai';
                llm.model     = cfg.llm.model;
                llm.chat      = chat;
                llm.available = true;
                return

            case 'azure'
                if isempty(which('azureChat'))
                    error('connectLLM:noAddon', 'azureChat not found.');
                end
                chat = azureChat('Temperature', cfg.llm.temperature);
                llm.backend   = 'azure';
                llm.model     = cfg.llm.model;
                llm.chat      = chat;
                llm.available = true;
                return
        end
    catch ME
        llm.lastError = ME.message;
        if cfg.io.verbose
            fprintf('  connectLLM: backend "%s" unavailable (%s)\n', b, ME.message);
        end
    end
end

if cfg.io.verbose
    fprintf(['  connectLLM: using the deterministic TEMPLATE backend. ', ...
             'Explanations will still be generated and the whole pipeline ', ...
             'will run; only the natural-language quality differs.\n']);
end
end
