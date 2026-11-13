function root = setupPaths()
%SETUPPATHS  Add every project folder to the MATLAB path.
%
%   ROOT = SETUPPATHS() returns the project root. Safe to call repeatedly.
%   Call this before anything else in a fresh MATLAB session.

root = fileparts(mfilename('fullpath'));
folders = {'config','core','environment','training','policy','shap','llm', ...
           'evaluation','dashboard','utils','tests','transfer'};
for i = 1:numel(folders)
    p = fullfile(root, folders{i});
    if exist(p,'dir')
        addpath(p);
    end
end
if ~exist(fullfile(root,'results'),'dir')
    mkdir(fullfile(root,'results'));
end
end
