import java.io.*;
import java.lang.reflect.Array;
import java.util.*;

/**
 * implemented trigram hidden Markov chain model with deleted interpolation using leave-one-out cross validation
 * refer to write-up for explanations and references
 * Viterbi Tagger
 *
 * @author Nour Hayek, Jack Zhang
 * February 27th, 2020
 */

public class ViterbiTaggerEC {

    public HashMap<String, HashMap<String, Double>> bigramTransCounts;  // transition counts for bigrams
    public HashMap<ArrayList<String>, HashMap<String, Double>> trigramTransCounts;  // transition counts for trigrams
    public HashMap<String, HashMap<String, Double>> bigramTransScores; // THe transition probability scores
    public HashMap<ArrayList<String>, HashMap<String, Double>> trigramTransScores; // THe transition probability scores
    public HashMap<String, HashMap<String, Double>> obsScores;  // the observation probability scores
    public double unknownScoreLog = -100.0;                     // the default scorer for an word not in the observation scores
    public double bigramWeight;                                 // interpolation weight for bigram
    public double trigramWeight;                                // interpolation weight for trigram

    /**
     * instantiate ViterbiTagger with files to train the model
     *
     * @param trainSentencesFileName name of file with sentences for training
     * @param trainTagsFileName      name of the file with corresponding tags
     */
    public ViterbiTaggerEC(String trainSentencesFileName, String trainTagsFileName) {

        try {
            bigramTrainingModel(trainSentencesFileName, trainTagsFileName);
            trigramTrainingModel(trainSentencesFileName, trainTagsFileName);
            calculateInterpolationWeights();
        } catch (IOException e) {
            System.err.println("Something went wrong while training model");
        }
    }

    /**
     * trains with bigram model
     * we condition the probability of a tag only on the previous tags
     *
     * @param trainSentencesFileName name of file with sentences for training
     * @param trainTagsFileName      name of the file with corresponding tags
     */
    public void bigramTrainingModel(String trainSentencesFileName, String trainTagsFileName) throws IOException {

        // read the files
        BufferedReader sentencesInput = new BufferedReader(new FileReader(trainSentencesFileName));
        BufferedReader tagsInput = new BufferedReader((new FileReader(trainTagsFileName)));

        // initializes transition scores and observation scores
        bigramTransScores = new HashMap<>();
        obsScores = new HashMap<>();

        //initializes sentence line and tag line from the files
        String currentSentenceLine;
        String currentTagLine;


        // adds the transition score for "start"
        bigramTransScores.put("start", new HashMap<>());

        // updates the count for transition scores and observation scores line by line
        while ((currentSentenceLine = sentencesInput.readLine()) != null) {

            currentTagLine = tagsInput.readLine();

            // initializes array of words and tags for the current line
            List<String> words = new ArrayList<>(Arrays.asList(currentSentenceLine.split(" ")));
            List<String> tags = new ArrayList<>(Arrays.asList((currentTagLine.split(" "))));

            // update transition scores for "start" with the first tag of the line
            if (!bigramTransScores.get("start").containsKey(tags.get(0))) {
                bigramTransScores.get("start").put(tags.get(0), 1.0);
            } else {
                bigramTransScores.get("start").put(tags.get(0), bigramTransScores.get("start").get(tags.get(0)) + 1.0);
            }

            // updates the transition scores for the remaining tags
            for (int i = 0; i < tags.size() - 1; i++) {
                // adds the tag to the map if it is not already in the key set
                if (!bigramTransScores.containsKey(tags.get(i))) {
                    HashMap<String, Double> newTransitionMap = new HashMap<>();
                    newTransitionMap.put(tags.get(i + 1), 1.0);
                    bigramTransScores.put(tags.get(i), newTransitionMap);
                    // updates the corresponding transition map if the tag is already in the key set
                } else {
                    //increments the score by one if the next tag is already in the map
                    if (bigramTransScores.get(tags.get(i)).containsKey(tags.get(i + 1))) {
                        bigramTransScores.get(tags.get(i)).put(tags.get(i + 1), bigramTransScores.get(tags.get(i)).get(tags.get(i + 1)) + 1);
                        // adds the next tag to the map if it is not already in it
                    } else {
                        bigramTransScores.get(tags.get(i)).put(tags.get(i + 1), 1.0);
                    }
                }
            }

            // updates the observation scores line by line
            for (int i = 0; i < words.size(); i++) {
                // adds the tag to the transition scores map if it's not already in the key set
                if (!obsScores.containsKey(tags.get(i))) {
                    HashMap<String, Double> newWordMap = new HashMap<>();
                    newWordMap.put(words.get(i), 1.0);
                    obsScores.put(tags.get(i), newWordMap);
                    // updates the map for the tag if the tag is already in the key set
                } else {
                    // increments the count by 1 if the word is already in the tag's map
                    if (obsScores.get(tags.get(i)).containsKey(words.get(i))) {
                        obsScores.get(tags.get(i)).put(words.get(i), obsScores.get(tags.get(i)).get(words.get(i)) + 1);
                        // creates a new map if with the word is the word is not in the tag's map
                    } else {
                        obsScores.get(tags.get(i)).put(words.get(i), 1.0);
                    }
                }
            }
        }

        // closes the files
        sentencesInput.close();
        tagsInput.close();

        bigramTransCounts = new HashMap<>();
        for (String tag : bigramTransScores.keySet()) {
            bigramTransCounts.put(tag, new HashMap<>());
            for (String nextTag : bigramTransScores.get(tag).keySet()) {
                Double currCount = bigramTransScores.get(tag).get(nextTag);
                bigramTransCounts.get(tag).put(nextTag, currCount);
            }
        }

        // normalizes the transition scores and observation scores
        for (String tag : bigramTransScores.keySet()) {
            // initializes the total count
            int totalTagCount = 0;

            // adds the tag counts iteratively
            for (String nextTag : bigramTransScores.get(tag).keySet()) {
                totalTagCount += bigramTransScores.get(tag).get(nextTag);
            }

            // divides each count by the corresponding total
            for (String nextTag : bigramTransScores.get(tag).keySet()) {
                bigramTransScores.get(tag).put(nextTag, Math.log(bigramTransScores.get(tag).get(nextTag) / totalTagCount));
            }
        }

        // normalizes the observation scores and observation scores
        for (String tag : obsScores.keySet()) {

            // initializes the total count
            int totalWordCount = 0;

            // adds the word counts iteratively
            for (String word : obsScores.get(tag).keySet()) {
                totalWordCount += obsScores.get(tag).get(word);
            }

            // divides each count by the corresponding total
            for (String word : obsScores.get(tag).keySet()) {
                obsScores.get(tag).put(word, Math.log(obsScores.get(tag).get(word) / totalWordCount));
            }
        }
    }

    /**
     * trains with trigram model
     * we condition the probability of a tag on the previous TWO tags
     *
     * @param trainSentencesFileName the file name of the training sentences
     * @param trainTagsFileName      the file name of the training tags
     */
    public void trigramTrainingModel(String trainSentencesFileName, String trainTagsFileName) throws IOException {

        // read the files
        BufferedReader sentencesInput = new BufferedReader(new FileReader(trainSentencesFileName));
        BufferedReader tagsInput = new BufferedReader((new FileReader(trainTagsFileName)));

        // initializes transition scores and observation scores
        trigramTransScores = new HashMap<>();
        obsScores = new HashMap<>();

        //initializes sentence line and tag line from the files
        String currentSentenceLine;
        String currentTagLine;


        // adds the transition score for [start, start]
        ArrayList<String> initialList = new ArrayList<>();
        initialList.add(0, "start");
        initialList.add(1, "start");
        trigramTransScores.put(initialList, new HashMap<>());

        int j = 0;
        // updates the count for transition scores and observation scores line by line
        while ((currentSentenceLine = sentencesInput.readLine()) != null) {


            currentTagLine = tagsInput.readLine();

            // initializes array of words and tags for the current line
            List<String> words = new ArrayList<>(Arrays.asList(currentSentenceLine.split(" ")));
            List<String> tags = new ArrayList<>(Arrays.asList((currentTagLine.split(" "))));

            // update transition scores for [start, start] with the first tag of the line
            if (!trigramTransScores.get(initialList).containsKey(tags.get(0))) {
                trigramTransScores.get(initialList).put(tags.get(0), 1.0);
            } else {
                trigramTransScores.get(initialList).put(tags.get(0), trigramTransScores.get(initialList).get(tags.get(0)) + 1.0);
            }
            trigramTransScores.get(initialList);

            // updates the transition scores for the remaining tags
            for (int i = 0; i < tags.size() - 1; i++) {
                // adds the tag to the map if it is not already in the key set

                ArrayList<String> currentTags = new ArrayList<>();
                if (i == 0) {
                    currentTags.add(0, "start");
                    currentTags.add(1, tags.get(i));
                } else {
                    currentTags.add(0, tags.get(i - 1));
                    currentTags.add(1, tags.get(i));
                }

                if (!trigramTransScores.containsKey(currentTags)) {
                    HashMap<String, Double> newTransitionMap = new HashMap<>();
                    newTransitionMap.put(tags.get(i + 1), 1.0);
                    trigramTransScores.put(currentTags, newTransitionMap);
                    // updates the corresponding transition map if the tag is already in the key set
                } else {
                    //increments the score by one if the next tag is already in the map
                    if (trigramTransScores.get(currentTags).containsKey(tags.get(i + 1))) {
                        trigramTransScores.get(currentTags).put(tags.get(i + 1), trigramTransScores.get(currentTags).get(tags.get(i + 1)) + 1);
                        // adds the next tag to the map if it is not already in it
                    } else {
                        trigramTransScores.get(currentTags).put(tags.get(i + 1), 1.0);
                    }
                }
            }

            // updates the observation scores line by line
            for (int i = 0; i < words.size(); i++) {
                // adds the tag to the transition scores map if it's not already in the key set
                if (!obsScores.containsKey(tags.get(i))) {
                    HashMap<String, Double> newWordMap = new HashMap<>();
                    newWordMap.put(words.get(i), 1.0);
                    obsScores.put(tags.get(i), newWordMap);
                    // updates the map for the tag if the tag is already in the key set
                } else {
                    // increments the count by 1 if the word is already in the tag's map
                    if (obsScores.get(tags.get(i)).containsKey(words.get(i))) {
                        obsScores.get(tags.get(i)).put(words.get(i), obsScores.get(tags.get(i)).get(words.get(i)) + 1);
                        // creates a new map if with the word is the word is not in the tag's map
                    } else {
                        obsScores.get(tags.get(i)).put(words.get(i), 1.0);
                    }
                }
            }
        }

        // closes the files
        sentencesInput.close();
        tagsInput.close();

        trigramTransCounts = new HashMap<>();
        for (ArrayList<String> tags : trigramTransScores.keySet()) {
            trigramTransCounts.put(tags, new HashMap<>());
            for (String nextTag : trigramTransScores.get(tags).keySet()) {
                Double currCount = trigramTransScores.get(tags).get(nextTag);
                trigramTransCounts.get(tags).put(nextTag, currCount);
            }
        }

        // normalizes the transition scores and observation scores
        for (ArrayList tags : trigramTransScores.keySet()) {
            // initializes the total count
            int totalTagCount = 0;

            // adds the tag counts iteratively
            for (String nextTag : trigramTransScores.get(tags).keySet()) {
                totalTagCount += trigramTransScores.get(tags).get(nextTag);
            }

            // divides each count by the corresponding total
            for (String nextTag : trigramTransScores.get(tags).keySet()) {
                trigramTransScores.get(tags).put(nextTag, trigramTransScores.get(tags).get(nextTag) / totalTagCount);

            }
        }

        // normalizes the observation scores and observation scores
        for (
                String tag : obsScores.keySet()) {

            // initializes the total count
            int totalWordCount = 0;

            // adds the word counts iteratively
            for (String word : obsScores.get(tag).keySet()) {
                totalWordCount += obsScores.get(tag).get(word);
            }

            // divides each count by the corresponding total
            for (String word : obsScores.get(tag).keySet()) {
                obsScores.get(tag).put(word, Math.log(obsScores.get(tag).get(word) / totalWordCount));
            }
        }
    }

    /**
     * tests the model with the test sentences and corresponding tags
     * write a file for the tags
     *
     * @param testSentencesFileName the file name of the testing sentences
     * @param resultFileName        the file name of the testing tags
     */
    public void testingModel(String testSentencesFileName, String resultFileName) throws IOException {

        // opens the file to read
        BufferedReader testInput = new BufferedReader(new FileReader(testSentencesFileName));
        // opens the file to write
        BufferedWriter result = new BufferedWriter(new FileWriter(resultFileName));

        // initializes the current line
        String currentLine;

        // reads each line and calls the Viterbi decoding method
        while ((currentLine = testInput.readLine()) != null) {


            List<String> currentTagList = viterbiDecoding(currentLine);

            // writes in the result file the tags from decoding
            String tagLine = "";
            for (int i = 0; i < currentTagList.size(); i++) tagLine += currentTagList.get(i) + " ";
            tagLine += "\n";
            result.write(tagLine);
        }

        // closes the files
        testInput.close();
        result.close();
    }

    /**
     * tests the model with the test sentences and corresponding tags
     * write a file for the tags
     *
     * @param resultFileName      the file name of the testing tags
     * @param correctTagsFileName the file name of the correct tags
     * @return Double the accuracy ratio
     */
    public Double calculateAccuracy(String resultFileName, String correctTagsFileName) throws IOException {

        // initializes the number of tags correct and the total tags in the result file
        int numCorrect = 0;
        int total = 0;

        // opens the files
        BufferedReader result = new BufferedReader(new FileReader(resultFileName));
        BufferedReader correctTags = new BufferedReader(new FileReader(correctTagsFileName));

        // initializes the lines from both files
        String currentResultLine;
        String currentTagLine;


        // iteratively count the number of tags correct by comparing with the corresponding correct tags
        while ((currentResultLine = result.readLine()) != null) {
            currentTagLine = correctTags.readLine();

            List<String> resultList = new ArrayList<>(Arrays.asList(currentResultLine.split(" ")));
            List<String> tagList = new ArrayList<>(Arrays.asList((currentTagLine.split(" "))));
            for (int i = 0; i < resultList.size(); i++) {
                if (resultList.get(i).equals(tagList.get(i))) numCorrect += 1;
                total += 1;
            }
        }

        // closes the files
        result.close();
        correctTags.close();

        // outputs the results
        System.out.println("The POS tagger identified " + numCorrect + " out of " + total + " tags correctly.");
        double accuracyRatio = ((double) numCorrect) / ((double) total);
        System.out.println("The accuracy is: " + (accuracyRatio) * 100 + "%");

        // returns the accuracy ratio
        return accuracyRatio;

    }

    /**
     * viterbi decoding
     *
     * @param line the string to be decoded to get the tags
     * @return List<String> the list of decoded tags
     */
    public List<String> viterbiDecoding(String line) throws NullPointerException {

        // lowercase all the lines
        line = line.toLowerCase();
        // initializes the word list by splitting with spaces


        List<String> wordsList = new ArrayList<>(Arrays.asList(line.split(" ")));
        // initializes the back trace map
        List<HashMap<ArrayList<String>, ArrayList<String>>> backTrace = new ArrayList<>();

        // initializes the set of current states and current scores (with [start, start] as the first state)
        Set<ArrayList<String>> currTrigramStates = new HashSet<>();
        ArrayList<String> initialStatePair = new ArrayList<>();
        initialStatePair.add(0, "start");
        initialStatePair.add(1, "start");
        currTrigramStates.add(initialStatePair);

        // the state is a pair of tags
        Map<ArrayList<String>, Double> currTrigramScores = new HashMap<>();
        currTrigramScores.put(initialStatePair, 0.0);

        // iterates through each word in the line to get the next states
        for (int i = 0; i < wordsList.size(); i++) {
            // initializes the next states and scores
            Set<ArrayList<String>> nextTrigramStates = new HashSet<>();
            Map<ArrayList<String>, Double> nextTrigramScores = new HashMap<>();

            // updates the next scores based on the transition and observation scores
            for (ArrayList<String> currStatePair : currTrigramStates) {

                // if trigram exists for the current tag pair
                boolean trigramExists = true;
                // if the current state is not in the transition scores, we skip it
                if (!trigramTransScores.containsKey(currStatePair)) {
                    trigramExists = false;
                }

                // consider trigram score when calculating the next score
                if (trigramExists) {
                    // adds the next states to the next states set
                    for (String nextState : trigramTransScores.get(currStatePair).keySet()) {
                        ArrayList<String> nextStatePair = new ArrayList<>();
                        nextStatePair.add(0, currStatePair.get(1));
                        nextStatePair.add(1, nextState);
                        nextTrigramStates.add(nextStatePair);

                        Double nextScore;

                        // if the observation score contains the word, we update according to Viterbi forward propagation
                        if (obsScores.get(nextState).containsKey(wordsList.get(i))) {
                            double weightedTransScore;
                            weightedTransScore = (bigramWeight) * bigramTransScores.get(currStatePair.get(1)).get(nextState) + (trigramWeight) * trigramTransScores.get(currStatePair).get(nextState);
                            nextScore = currTrigramScores.get(currStatePair) + weightedTransScore + obsScores.get(nextState).get(wordsList.get(i));
                            // if the word has not bee encountered in training, we default the observation score
                        } else {
                            // calculates weighted average of the bigram and trigram scores
                            double weightedTransScore;
                            weightedTransScore = (bigramWeight) * bigramTransScores.get(currStatePair.get(1)).get(nextState) + (trigramWeight) * trigramTransScores.get(currStatePair).get(nextState);
                            nextScore = currTrigramScores.get(currStatePair) + weightedTransScore + unknownScoreLog;
                        }

                        // if the next state has not been encountered yet in this line or we found a smaller next score, we update accordingly
                        if (!nextTrigramScores.containsKey(nextStatePair) || nextScore > nextTrigramScores.get(nextStatePair)) {
                            // adds the back pointers to each state
                            nextTrigramScores.put(nextStatePair, nextScore);
                            if (backTrace.size() < i + 1) {
                                backTrace.add(i, new HashMap<>());
                                backTrace.get(i).put(nextStatePair, currStatePair);
                            } else {
                                backTrace.get(i).put(nextStatePair, currStatePair);
                            }
                        }
                    }
                    // does not consider trigram score if it does not exist for the tag pair
                } else {
                    if (!bigramTransScores.containsKey(currStatePair.get(1))) continue;
                    for (String nextState : bigramTransScores.get(currStatePair.get(1)).keySet()) {
                        ArrayList<String> nextStatePair = new ArrayList<>();
                        nextStatePair.add(0, currStatePair.get(1));
                        nextStatePair.add(1, nextState);
                        nextTrigramStates.add(nextStatePair);

                        Double nextScore;
                        // if the observation score contains the word, we update according to Viterbi forward propagation
                        if (obsScores.get(nextState).containsKey(wordsList.get(i))) {
                            double weightedTransScore;
                            // only considers the bigram score
                            weightedTransScore = bigramTransScores.get(currStatePair.get(1)).get(nextState);
                            nextScore = currTrigramScores.get(currStatePair) + weightedTransScore + obsScores.get(nextState).get(wordsList.get(i));
                            // if the word has not bee encountered in training, we default the observation score
                        } else {
                            double weightedTransScore;
                            weightedTransScore = bigramTransScores.get(currStatePair.get(1)).get(nextState);

                            nextScore = currTrigramScores.get(currStatePair) + weightedTransScore + unknownScoreLog;
                        }

                        // if the next state has not been encountered yet in this line or we found a smaller next score, we update accordingly
                        if (!nextTrigramScores.containsKey(nextStatePair) || nextScore > nextTrigramScores.get(nextStatePair)) {
                            // adds the back pointers to each state
                            nextTrigramScores.put(nextStatePair, nextScore);
                            if (backTrace.size() < i + 1) {
                                backTrace.add(i, new HashMap<>());
                                backTrace.get(i).put(nextStatePair, currStatePair);
                            } else {
                                backTrace.get(i).put(nextStatePair, currStatePair);
                            }
                        }
                    }

                }
            }
            // updates the current states for next iteration
            currTrigramStates = nextTrigramStates;
            currTrigramScores = nextTrigramScores;
        }

        // initializes the linked list of tags
        List<String> tags = new LinkedList();
        // initializes the last tag as the one with the highest score
        ArrayList<String> bestStates = new ArrayList<>();
        double highestScore = (-1) * Double.MAX_VALUE;

        for (ArrayList<String> statePair : currTrigramScores.keySet()) {
            if (currTrigramScores.get(statePair) > highestScore) {
                bestStates = statePair;
                highestScore = currTrigramScores.get(statePair);
            }
        }

        tags.add(0, bestStates.get(1));

        // trace backward in the line with the back pointers in the back trace map
        for (int i = wordsList.size() - 1; i > 0; i--) {
            bestStates = backTrace.get(i).get(bestStates);
            tags.add(0, bestStates.get(1));
        }

        // returns the decoded tags in a list
        return tags;
    }

    /**
     * calculates weights for trigram and bigram scores based on maximum likelihood
     * see extra credit write-up for explanation
     *
     */
    public void calculateInterpolationWeights() {
        // initialize weights
        double bigramW = 0.0;
        double trigramW = 0.0;

        // initialize count
        double c1 = 0;
        double c2 = 0;

        // leave-one-out cross validation
        for (ArrayList<String> tags : trigramTransCounts.keySet()) {
            for (String nextTag : trigramTransCounts.get(tags).keySet()) {
                Double deletedCountTrigram = trigramTransCounts.get(tags).get(nextTag);

                if (deletedCountTrigram > 0) {
                    try {
                        c1 = (deletedCountTrigram - 1) / (bigramTransCounts.get(tags.get(0)).get(tags.get(1)) - 1);
                    } catch (ArithmeticException e) {
                        c1 = 0;
                        // [start, start] in the trigram key set is not in the bigram key set
                    } catch (NullPointerException e) {
                        c1 = 0;
                    }

                    try {

                        double currSumBigram = 0;

                        for (double cnt : bigramTransCounts.get(tags.get(0)).values()) {
                            currSumBigram += cnt;
                        }

                        c2 = (bigramTransCounts.get(tags.get(0)).get(tags.get(1)) - 1) / (currSumBigram - 1);

                    } catch (ArithmeticException e) {
                        c2 = 0;
                        // [start, start] in the trigram key set is not in the bigram key set
                    } catch (NullPointerException e) {
                        c2 = 0;
                    }
                }
                // update weights based on maximum likelihood
                if (c1 >= c2) trigramW += deletedCountTrigram;
                else bigramW += deletedCountTrigram;
            }
        }

        // normalize weights
        double totalWeight = trigramW + bigramW;
        double bigramWeightNormalized = bigramW / totalWeight;
        double trigramWeightNormalized = trigramW / totalWeight;

        this.bigramWeight = bigramWeightNormalized;
        this.trigramWeight = trigramWeightNormalized;

        System.out.println("bigram weight: " + bigramWeight);
        System.out.println("trigram weight: " + trigramWeight);
    }

    /**
     * console based tagging
     */
    public void consoleBasedTagger() {
        Scanner in = new Scanner(System.in);

        // takes inputs from user until the user quits
        while (true) {
            System.out.println("Please enter a sentence to get tags (enter \"q\" to quit game)");
            System.out.print("> ");
            String line = in.nextLine();
            // if user inputs "q", the tagger ends
            if (line.equals("q")) return;

            // calls Viterbi decoding on the provided line
            List<String> tagList = viterbiDecoding(line);

            // outputs the decoded tags
            String tagLine = "";
            for (int i = 0; i < tagList.size(); i++) tagLine += tagList.get(i) + " ";
            tagLine += "\n";
            System.out.println(tagLine);
        }
    }

    // 3 tests for the Viterbi Tagger class
    public static void main(String[] args) {

//        // hard coded transition scores from programming drill
//        System.out.println("Beginning test 0...");
//        ViterbiTaggerEC test0 = new ViterbiTaggerEC();
//
//        System.out.println("Inserting first HMM graph...");
//        HashMap<String, HashMap<String, Double>> trigramTransScores = new HashMap<>();
//
//        HashMap<String, Double> scores = new HashMap<>();
//        scores.put("NP", (double) (3 / 10));
//        scores.put("N", (double) (7 / 10));
//        trigramTransScores.put("start", scores);
//
//        scores = new HashMap<>();
//        scores.put("V", (double) (8 / 10));
//        scores.put("CNJ", (double) (2 / 10));
//        trigramTransScores.put("NP", scores);
//
//        scores = new HashMap<>();
//        scores.put("N", (double) (4 / 6));
//        scores.put("NP", (double) (2 / 6));
//        trigramTransScores.put("CNJ", scores);
//
//        scores = new HashMap<>();
//        scores.put("CNJ", (double) (2 / 10));
//        scores.put("V", (double) (8 / 10));
//        trigramTransScores.put("N", scores);
//
//        scores = new HashMap<>();
//        scores.put("NP", (double) (4 / 8));
//        scores.put("N", (double) (4 / 8));
//        trigramTransScores.put("V", scores);
//
//        // manually sets the transition scores
//        test0.setTransScores(trigramTransScores);
//
//
//        // hard coded observation scores from programming drill
//        HashMap<String, HashMap<String, Double>> obsScores = new HashMap<>();
//
//        scores = new HashMap<>();
//        scores.put("Chase", (double) 1);
//        obsScores.put("NP", scores);
//
//        scores = new HashMap<>();
//        scores.put("cat", (double) (4 / 10));
//        scores.put("dog", (double) (4 / 10));
//        scores.put("watch", (double) (2 / 10));
//        obsScores.put("N", scores);
//
//        scores = new HashMap<>();
//        scores.put("and", (double) (1));
//        obsScores.put("CNJ", scores);
//
//        scores = new HashMap<>();
//        scores.put("get", (double) (1 / 10));
//        scores.put("chase", (double) (3 / 10));
//        scores.put("watch", (double) (6 / 10));
//        obsScores.put("V", scores);
//
//        // manually sets the observation scores
//        test0.setobsScores(obsScores);
//
//
//        // testing the hard coded HMM
//        try {
//            System.out.println("Testing on sentence: \"cat watch chase and dog\"");
//            System.out.println("The corresponding tags are: " + test0.viterbiDecoding("cat watch chase and dog"));
//        } catch (NullPointerException e) {
//            System.err.println(e.getMessage());
//        }
//
//        System.out.println("\nInserting second HMM graph...");
//        trigramTransScores = new HashMap<>();
//
//        scores = new HashMap<>();
//        scores.put("NP", (double) (1 / 2));
//        scores.put("VG", (double) (1 / 2));
//        trigramTransScores.put("start", scores);
//
//        scores = new HashMap<>();
//        scores.put("ADJ", (double) (1));
//        trigramTransScores.put("NP", scores);
//
//        scores = new HashMap<>();
//        scores.put("V", (double) (1));
//        trigramTransScores.put("VG", scores);
//
//        scores = new HashMap<>();
//        scores.put("N", (double) (1 / 2));
//        scores.put("V", (double) (1 / 2));
//        trigramTransScores.put("ADJ", scores);
//
//        scores = new HashMap<>();
//        scores.put("ADJ", (double) (1));
//        trigramTransScores.put("V", scores);
//
//        scores = new HashMap<>();
//        scores.put("VG", (double) (1 / 2));
//        scores.put("DET", (double) (1 / 2));
//        trigramTransScores.put("DET", scores);
//
//        // manually sets the transition scores
//        test0.setTransScores(trigramTransScores);
//
//
//        // hard coded observation scores from programming drill
//        obsScores = new HashMap<>();
//
//        scores = new HashMap<>();
//        scores.put("I", (double) 1);
//        obsScores.put("NP", scores);
//
//        scores = new HashMap<>();
//        scores.put("swimming", (double) (1));
//        obsScores.put("VG", scores);
//
//        scores = new HashMap<>();
//        scores.put("really", (double) (1 / 2));
//        scores.put("fun", (double) (1 / 2));
//        obsScores.put("ADJ", scores);
//
//        scores = new HashMap<>();
//        scores.put("enjoy", (double) (1 / 2));
//        scores.put("is", (double) (1 / 2));
//        obsScores.put("V", scores);
//
//        scores = new HashMap<>();
//        scores.put("a", (double) (1));
//        obsScores.put("DET", scores);
//
//        scores = new HashMap<>();
//        scores.put("sport", (double) (1));
//        obsScores.put("N", scores);
//
//        // manually sets the observation scores
//        test0.setobsScores(obsScores);
//
//
//        // testing the hard coded HMM
//        try {
//            System.out.println("Testing on sentence: \"Swimming is a really fun sport\"");
//            System.out.println("The corresponding tags are: " + test0.viterbiDecoding("Swimming is a really fun sport"));
//        } catch (NullPointerException e) {
//            System.err.println(e.getMessage());
//        }
//
        // tests with the simple training and testing data provided
        System.out.println("\nBeginning test 1...");
        System.out.println("Training with the simple sentences");
        ViterbiTaggerEC test1 = new ViterbiTaggerEC("texts/simple-train-sentences.txt", "texts/simple-train-tags.txt");

        System.out.println("Testing on simple test sentences");

        try {
            test1.testingModel("texts/simple-test-sentences.txt", "texts/simple-test-tags-result-EC.txt");
        } catch (IOException e) {
            System.err.println("Something went wrong while testing model");
        }
//
//        catch (NullPointerException e) {
//            System.err.println(e.getMessage());
//        }

        try {
            test1.calculateAccuracy("texts/simple-test-tags-result-EC.txt", "texts/simple-test-tags.txt");
        } catch (IOException e) {
            System.err.println("Something went wrong while calculating accuracy");
        }


        // testing with the Brown Corpus
        System.out.println("\nBeginning test 2...");
        System.out.println("Training with the Brown corpus");
        ViterbiTaggerEC test2 = new ViterbiTaggerEC("texts/brown-train-sentences.txt", "texts/brown-train-tags.txt");

        System.out.println("Testing on Brown test sentences");
        try {
            test2.testingModel("texts/brown-test-sentences.txt", "texts/brown-test-tags-result-EC.txt");
        } catch (IOException e) {
            System.err.println("Something went wrong while testing model");
        } catch (NullPointerException e) {
            System.err.println(e.getMessage());
        }

        try {
            test2.calculateAccuracy("texts/brown-test-tags-result-EC.txt", "texts/brown-test-tags.txt");
        } catch (IOException e) {
            System.err.println("Something went wrong while calculating accuracy");
        }
//
//        // console based tagging that takes user input
//        System.out.println("\nBeginning console-based tagging...");
//        test2.consoleBasedTagger();
//
//        System.out.println("\nThe end");
    }
}
