function create_profile_data

sizes = {...
    [ 10,  8,  6, 4],...
    [ 20, 16, 12],...
    [ 1000, 3, 3],...
    };

rank = 3;

num_tensors = 3;

filesuffix = 'tns';

filetype = 'tensor_dense_continuous';
for i = 1:length(sizes)
    for j = 1:num_tensors
        rng(j);
        info = create_problem('Size',sizes{i},'Num_Factors',rank);
        s = strrep(tt_size2str(info.Data.size),' x ','_');
        f = sprintf('%s_size_%s_rng_%d.%s',filetype,s,j,filesuffix);
        export_data(info.Data,f);
    end
end

filetype = 'tensor_dense_integer';
for i = 1:length(sizes)
    for j = 1:num_tensors
        rng(10*j);
        M = get_ktensor(sizes{i},rank);
        info = create_problem('Soln',M,'Sparse_Generation', 0);
        %info.Data = tensor(abs(floor(10*info.Data.data)));
        info.Data = tensor(floor(abs(1./sqrt(max(info.Data.data,[],'all'))*info.Data.data))+1);
        s = strrep(tt_size2str(info.Data.size),' x ','_');
        f = sprintf('%s_size_%s_rng_%d.%s',filetype,s,j,filesuffix);
        export_data(info.Data,f,'fmt_data','%d');
    end
end

filetype = 'sptensor_sparse_continuous';
for i = 1:length(sizes)
    for j = 1:num_tensors
        rng(j);
        info = create_problem('Size',sizes{i},'Num_Factors',rank);
        ind = randperm(prod(size(info.Data)));
        ind = ind(1:floor(0.85*prod(size(info.Data))));
        info.Data(ind') = 0;
        info.Data = sptensor(info.Data);
        s = strrep(tt_size2str(info.Data.size),' x ','_');
        f = sprintf('%s_size_%s_rng_%d.%s',filetype,s,j,filesuffix);
        export_data(info.Data,f);
    end
end

filetype = 'sptensor_sparse_integer';
for i = 1:length(sizes)
    for j = 1:num_tensors
        rng(10*j);
        M = get_ktensor(sizes{i},rank);
        info = create_problem('Soln',M,'Sparse_Generation', 0);
        %info.Data = tensor(abs(floor(10*info.Data.data)));
        info.Data = tensor(floor(abs(1./sqrt(max(info.Data.data,[],'all'))*info.Data.data))+1);
        ind = randperm(prod(size(info.Data)));
        ind = ind(1:floor(0.85*prod(size(info.Data))));
        info.Data(ind') = 0;
        info.Data = sptensor(info.Data);
        s = strrep(tt_size2str(info.Data.size),' x ','_');
        f = sprintf('%s_size_%s_rng_%d.%s',filetype,s,j,filesuffix);
        export_data(info.Data,f,'fmt_data','%d');
    end
end


end % function

%% helper function

function S = get_ktensor(sz,R)

A = cell(length(sz),1);
for n = 1:length(sz)
    A{n} = rand(sz(n), R);
    for r = 1:R
        p = randperm(sz(n));
        nbig = round( (1/R)*sz(n) );
        A{n}(p(1:nbig),r) = 100 * A{n}(p(1:nbig),r);
    end
end
lambda = rand(R,1);
S = ktensor(lambda, A);
S = normalize(S,'sort',1);

end