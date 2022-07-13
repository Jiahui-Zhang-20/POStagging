import java.io.*;
import java.util.*;

/**
 * POS Tagger
 *
 * @author Nour Hayek, Jack Zhang
 * February 27th, 2020
 */

public class PosTagger {

    public HashMap<String, HashMap<String, Double>> transScores; // THe transition probability scores
    public HashMap<String, HashMap<String, Double>> obsScores;  // the observation probability scores
    public double unknownScoreLog = -100.0;                     // the default scorer for an word not in the observation scores

    /**
     * instantiate ViterbiTagger without passing in file names (must manually set the transition and observation scores)
     */
    public PosTagger() {
    }
    // instantiate ViterbiTagger with files to train the model

    /**
     * instantiate ViterbiTagger with files to train the model
     *
     * @param trainSentencesFileName name of file with sentences for training
     * @param trainTagsFileName      name of the file with corresponding tags
     */
    public PosTagger(String trainSentencesFileName, String trainTagsFileName) {

        try {
            trainingModel(trainSentencesFileName, trainTagsFileName);
        } catch (IOException e) {
            System.err.println("Something went wrong while training model");
        }
    }

    /**
     * manually sets transition scores
     *
     * @param scores transition scores
     */
    public void setTransScores(HashMap<String, HashMap<String, Double>> scores) {
        this.transScores = scores;
    }

    /**
     * manually sets observation scores
     *
     * @param scores observation scores
     */
    public void setobsScores(HashMap<String, HashMap<String, Double>> scores) {
        this.obsScores = scores;
    }


    /**
     * trains the model with the training sentences and corresponding tags
     *
     * @param trainSentencesFileName the file name of the training sentences
     * @param trainTagsFileName      the file name of the training tags
     */
    public void trainingModel(String trainSentencesFileName, String trainTagsFileName) throws IOException {

        // read the files
        BufferedReader sentencesInput = new BufferedReader(new FileReader(trainSentencesFileName));
        BufferedReader tagsInput = new BufferedReader((new FileReader(trainTagsFileName)));

        // initializes transition scores and observation scores
        transScores = new HashMap<>();
        obsScores = new HashMap<>();

        //initializes sentence line and tag line from the files
        String currentSentenceLine;
        String currentTagLine;


        // adds the transition score for "start"
        transScores.put("start", new HashMap<>());

        // updates the count for transition scores and observation scores line by line
        while ((currentSentenceLine = sentencesInput.readLine()) != null) {

            currentTagLine = tagsInput.readLine();

            // initializes array of words and tags for the current line
            List<String> words = new ArrayList<>(Arrays.asList(currentSentenceLine.split(" ")));
            List<String> tags = new ArrayList<>(Arrays.asList((currentTagLine.split(" "))));

            // update transition scores for "start" with the first tag of the line
            if (!transScores.get("start").containsKey(tags.get(0))) {
                transScores.get("start").put(tags.get(0), 1.0);
            } else {
                transScores.get("start").put(tags.get(0), transScores.get("start").get(tags.get(0)) + 1.0);
            }

            // updates the transition scores for the remaining tags
            for (int i = 0; i < tags.size() - 1; i++) {
                // adds the tag to the map if it is not already in the key set
                if (!transScores.containsKey(tags.get(i))) {
                    HashMap<String, Double> newTransitionMap = new HashMap<>();
                    newTransitionMap.put(tags.get(i + 1), 1.0);
                    transScores.put(tags.get(i), newTransitionMap);
                    // updates the corresponding transition map if the tag is already in the key set
                } else {
                    //increments the score by one if the next tag is already in the map
                    if (transScores.get(tags.get(i)).containsKey(tags.get(i + 1))) {
                        transScores.get(tags.get(i)).put(tags.get(i + 1), transScores.get(tags.get(i)).get(tags.get(i + 1)) + 1);
                        // adds the next tag to the map if it is not already in it
                    } else {
                        transScores.get(tags.get(i)).put(tags.get(i + 1), 1.0);
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

        // normalizes the transition scores and observation scores
        for (String tag : transScores.keySet()) {
            // initializes the total count
            int totalTagCount = 0;

            // adds the tag counts iteratively
            for (String nextTag : transScores.get(tag).keySet()) {
                totalTagCount += transScores.get(tag).get(nextTag);
            }

            // divides each count by the corresponding total
            for (String nextTag : transScores.get(tag).keySet()) {
                transScores.get(tag).put(nextTag, Math.log(transScores.get(tag).get(nextTag) / totalTagCount));
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

        // returns if we do not have the transition scores or observation scores
        // they can be manually created and updated with the corresponding setter functions
        if (transScores == null || obsScores == null)
            throw new NullPointerException("Please provide the transition scores and/or observation scores");
        // lowercase all the lines
        line = line.toLowerCase();

        // initializes the word list by splitting with spaces
        List<String> wordsList = new ArrayList<>(Arrays.asList(line.split(" ")));
        // initializes the back trace map
        List<HashMap<String, String>> backTrace = new ArrayList<>();

        // initializes the set of current states and current scores (with "start" as the first state)
        Set<String> currStates = new HashSet<>();
        currStates.add("start");
        Map<String, Double> currScores = new TreeMap<>();
        currScores.put("start", 0.0);

        // iterates through each word in the line to get the next states
        for (int i = 0; i < wordsList.size(); i++) {
            // initializes the next states and scores
            Set<String> nextStates = new HashSet<>();
            Map<String, Double> nextScores = new HashMap<>();

            // updates the next scores based on the transition and observation scores
            for (String currState : currStates) {
                // if the current state is not in the transition scores, we skip it
                if (!transScores.containsKey(currState)) continue;

                // adds the next states to the next states set
                for (String nextState : transScores.get(currState).keySet()) {
                    nextStates.add(nextState);
                    Double nextScore;

                    // if the observation score contains the word, we update according to Viterbi forward propagation
                    if (obsScores.get(nextState).containsKey(wordsList.get(i))) {
                        nextScore = currScores.get(currState) + transScores.get(currState).get(nextState) + obsScores.get(nextState).get(wordsList.get(i));
                        // if the word has not bee encountered in training, we default the observation score
                    } else {
                        nextScore = currScores.get(currState) + transScores.get(currState).get(nextState) + unknownScoreLog;
                    }

                    // if the next state has not been encountered yet in this line or we found a smaller next score, we update accordingly
                    if (!nextScores.containsKey(nextState) || nextScore > nextScores.get(nextState)) {
                        // adds the back pointers to each state
                        nextScores.put(nextState, nextScore);
                        if (backTrace.size() < i + 1) {
                            backTrace.add(i, new HashMap<>());
                            backTrace.get(i).put(nextState, currState);
                        } else {
                            backTrace.get(i).put(nextState, currState);
                        }

                    }
                }
            }
            // updates the current states for next iteration
            currStates = nextStates;
            currScores = nextScores;
        }

        // initializes the linked list of tags
        List<String> tags = new LinkedList();
        // initializes the last tag as the one with the highest score
        String bestState = Collections.max(currScores.entrySet(), Map.Entry.comparingByValue()).getKey();
        tags.add(0, bestState);

        // trace backward in the line with the back pointers in the back trace map
        for (int i = wordsList.size() - 1; i > 0; i--) {
            bestState = backTrace.get(i).get(bestState);
            tags.add(0, bestState);
        }

        // returns the decoded tags in a list
        return tags;
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

        // hard coded transition scores from programming drill
        System.out.println("Beginning test 0...");
        PosTagger test0 = new PosTagger();

        System.out.println("Inserting first HMM graph...");
        HashMap<String, HashMap<String, Double>> transScores = new HashMap<>();

        HashMap<String, Double> scores = new HashMap<>();
        scores.put("NP", (double) (3 / 10));
        scores.put("N", (double) (7 / 10));
        transScores.put("start", scores);

        scores = new HashMap<>();
        scores.put("V", (double) (8 / 10));
        scores.put("CNJ", (double) (2 / 10));
        transScores.put("NP", scores);

        scores = new HashMap<>();
        scores.put("N", (double) (4 / 6));
        scores.put("NP", (double) (2 / 6));
        transScores.put("CNJ", scores);

        scores = new HashMap<>();
        scores.put("CNJ", (double) (2 / 10));
        scores.put("V", (double) (8 / 10));
        transScores.put("N", scores);

        scores = new HashMap<>();
        scores.put("NP", (double) (4 / 8));
        scores.put("N", (double) (4 / 8));
        transScores.put("V", scores);

        // manually sets the transition scores
        test0.setTransScores(transScores);


        // hard coded observation scores from programming drill
        HashMap<String, HashMap<String, Double>> obsScores = new HashMap<>();

        scores = new HashMap<>();
        scores.put("Chase", (double) 1);
        obsScores.put("NP", scores);

        scores = new HashMap<>();
        scores.put("cat", (double) (4 / 10));
        scores.put("dog", (double) (4 / 10));
        scores.put("watch", (double) (2 / 10));
        obsScores.put("N", scores);

        scores = new HashMap<>();
        scores.put("and", (double) (1));
        obsScores.put("CNJ", scores);

        scores = new HashMap<>();
        scores.put("get", (double) (1 / 10));
        scores.put("chase", (double) (3 / 10));
        scores.put("watch", (double) (6 / 10));
        obsScores.put("V", scores);

        // manually sets the observation scores
        test0.setobsScores(obsScores);


        // testing the hard coded HMM
        try {
            System.out.println("Testing on sentence: \"cat watch chase and dog\"");
            System.out.println("The corresponding tags are: " + test0.viterbiDecoding("cat watch chase and dog"));
        } catch (NullPointerException e) {
            System.err.println(e.getMessage());
        }

        System.out.println("\nInserting second HMM graph...");
        transScores = new HashMap<>();

        scores = new HashMap<>();
        scores.put("NP", (double) (1 / 2));
        scores.put("VG", (double) (1 / 2));
        transScores.put("start", scores);

        scores = new HashMap<>();
        scores.put("ADJ", (double) (1));
        transScores.put("NP", scores);

        scores = new HashMap<>();
        scores.put("V", (double) (1));
        transScores.put("VG", scores);

        scores = new HashMap<>();
        scores.put("N", (double) (1 / 2));
        scores.put("V", (double) (1 / 2));
        transScores.put("ADJ", scores);

        scores = new HashMap<>();
        scores.put("ADJ", (double) (1));
        transScores.put("V", scores);

        scores = new HashMap<>();
        scores.put("VG", (double) (1 / 2));
        scores.put("DET", (double) (1 / 2));
        transScores.put("DET", scores);

        // manually sets the transition scores
        test0.setTransScores(transScores);


        // hard coded observation scores from programming drill
        obsScores = new HashMap<>();

        scores = new HashMap<>();
        scores.put("I", (double) 1);
        obsScores.put("NP", scores);

        scores = new HashMap<>();
        scores.put("swimming", (double) (1));
        obsScores.put("VG", scores);

        scores = new HashMap<>();
        scores.put("really", (double) (1 / 2));
        scores.put("fun", (double) (1 / 2));
        obsScores.put("ADJ", scores);

        scores = new HashMap<>();
        scores.put("enjoy", (double) (1 / 2));
        scores.put("is", (double) (1 / 2));
        obsScores.put("V", scores);

        scores = new HashMap<>();
        scores.put("a", (double) (1));
        obsScores.put("DET", scores);

        scores = new HashMap<>();
        scores.put("sport", (double) (1));
        obsScores.put("N", scores);

        // manually sets the observation scores
        test0.setobsScores(obsScores);


        // testing the hard coded HMM
        try {
            System.out.println("Testing on sentence: \"Swimming is a really fun sport\"");
            System.out.println("The corresponding tags are: " + test0.viterbiDecoding("Swimming is a really fun sport"));
        } catch (NullPointerException e) {
            System.err.println(e.getMessage());
        }

        // tests with the simple training and testing data provdied
        System.out.println("\nBeginning test 1...");
        System.out.println("Training with the simple sentences");
        PosTagger test1 = new PosTagger("texts/simple-train-sentences.txt", "texts/simple-train-tags.txt");

        System.out.println("Testing on simple test sentences");

        try {
            test1.testingModel("texts/simple-test-sentences.txt", "texts/simple-test-tags-result.txt");
        } catch (IOException e) {
            System.err.println("Something went wrong while testing model");
        } catch (NullPointerException e) {
            System.err.println(e.getMessage());
        }

        try {
            test1.calculateAccuracy("texts/simple-test-tags-result.txt", "texts/simple-test-tags.txt");
        } catch (IOException e) {
            System.err.println("Something went wrong while calculating accuracy");
        }


        // testing with the Brown Corpus
        System.out.println("\nBeginning test 2...");
        System.out.println("Training with the Brown corpus");
        PosTagger test2 = new PosTagger("texts/brown-train-sentences.txt", "texts/brown-train-tags.txt");

        System.out.println("Testing on Brown test sentences");
        try {
            test2.testingModel("texts/brown-test-sentences.txt", "texts/brown-test-tags-result.txt");
        } catch (IOException e) {
            System.err.println("Something went wrong while testing model");
        } catch (NullPointerException e) {
            System.err.println(e.getMessage());
        }

        try {
            test2.calculateAccuracy("texts/brown-test-tags-result.txt", "texts/brown-test-tags.txt");
        } catch (IOException e) {
            System.err.println("Something went wrong while calculating accuracy");
        }

        // console based tagging that takes user input
        System.out.println("\nBeginning console-based tagging...");
        test2.consoleBasedTagger();

        System.out.println("\nThe end");
    }

}

