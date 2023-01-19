function mean_seq = compute_sequence_scores(all_scores)
    mean_seq = sum(all_scores([1:3 6 15 20 23 24])) + mean(all_scores(4:5)) + mean(all_scores(7:12)) ...
        + mean(all_scores(13:14)) + mean(all_scores(16:17)) + mean(all_scores(18:19)) + mean(all_scores(21:22));
    mean_seq = mean_seq / 14;
end
