package io.github.betterclient.ai;

import io.github.betterclient.ai.model.digit.DigitRecognitionModel;
import io.github.betterclient.ai.model.digit.Position;
import io.github.betterclient.ai.neural.NeuralLayer;
import io.github.betterclient.ai.neural.Neuron;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.math.BigDecimal;
import java.util.Map;

public class DebuggingMain {
    public static void main(String[] args) throws Exception {
        Map<Position, Boolean> mapof6 = Map.<Position, Boolean>ofEntries(Map.entry(new Position(0, 0), false), Map.entry(new Position(0, 1), false), Map.entry(new Position(0, 2), false), Map.entry(new Position(0, 3), false), Map.entry(new Position(0, 4), false), Map.entry(new Position(0, 5), false), Map.entry(new Position(0, 6), false), Map.entry(new Position(0, 7), false), Map.entry(new Position(0, 8), false), Map.entry(new Position(0, 9), false), Map.entry(new Position(0, 10), false), Map.entry(new Position(0, 11), false), Map.entry(new Position(0, 12), false), Map.entry(new Position(0, 13), false), Map.entry(new Position(0, 14), false), Map.entry(new Position(0, 15), false), Map.entry(new Position(1, 0), false), Map.entry(new Position(1, 1), false), Map.entry(new Position(1, 2), false), Map.entry(new Position(1, 3), false), Map.entry(new Position(1, 4), false), Map.entry(new Position(1, 5), false), Map.entry(new Position(1, 6), false), Map.entry(new Position(1, 7), false), Map.entry(new Position(1, 8), false), Map.entry(new Position(1, 9), false), Map.entry(new Position(1, 10), false), Map.entry(new Position(1, 11), false), Map.entry(new Position(1, 12), false), Map.entry(new Position(1, 13), false), Map.entry(new Position(1, 14), false), Map.entry(new Position(1, 15), false), Map.entry(new Position(2, 0), false), Map.entry(new Position(2, 1), false), Map.entry(new Position(2, 2), false), Map.entry(new Position(2, 3), false), Map.entry(new Position(2, 4), false), Map.entry(new Position(2, 5), false), Map.entry(new Position(2, 6), false), Map.entry(new Position(2, 7), false), Map.entry(new Position(2, 8), false), Map.entry(new Position(2, 9), false), Map.entry(new Position(2, 10), false), Map.entry(new Position(2, 11), false), Map.entry(new Position(2, 12), false), Map.entry(new Position(2, 13), false), Map.entry(new Position(2, 14), false), Map.entry(new Position(2, 15), false), Map.entry(new Position(3, 0), false), Map.entry(new Position(3, 1), false), Map.entry(new Position(3, 2), false), Map.entry(new Position(3, 3), false), Map.entry(new Position(3, 4), false), Map.entry(new Position(3, 5), false), Map.entry(new Position(3, 6), false), Map.entry(new Position(3, 7), false), Map.entry(new Position(3, 8), false), Map.entry(new Position(3, 9), false), Map.entry(new Position(3, 10), false), Map.entry(new Position(3, 11), true), Map.entry(new Position(3, 12), true), Map.entry(new Position(3, 13), false), Map.entry(new Position(3, 14), false), Map.entry(new Position(3, 15), false), Map.entry(new Position(4, 0), false), Map.entry(new Position(4, 1), false), Map.entry(new Position(4, 2), false), Map.entry(new Position(4, 3), false), Map.entry(new Position(4, 4), false), Map.entry(new Position(4, 5), true), Map.entry(new Position(4, 6), true), Map.entry(new Position(4, 7), true), Map.entry(new Position(4, 8), true), Map.entry(new Position(4, 9), true), Map.entry(new Position(4, 10), true), Map.entry(new Position(4, 11), true), Map.entry(new Position(4, 12), true), Map.entry(new Position(4, 13), true), Map.entry(new Position(4, 14), false), Map.entry(new Position(4, 15), false), Map.entry(new Position(5, 0), false), Map.entry(new Position(5, 1), false), Map.entry(new Position(5, 2), false), Map.entry(new Position(5, 3), true), Map.entry(new Position(5, 4), true), Map.entry(new Position(5, 5), false), Map.entry(new Position(5, 6), false), Map.entry(new Position(5, 7), true), Map.entry(new Position(5, 8), false), Map.entry(new Position(5, 9), false), Map.entry(new Position(5, 10), false), Map.entry(new Position(5, 11), false), Map.entry(new Position(5, 12), false), Map.entry(new Position(5, 13), true), Map.entry(new Position(5, 14), true), Map.entry(new Position(5, 15), false), Map.entry(new Position(6, 0), false), Map.entry(new Position(6, 1), false), Map.entry(new Position(6, 2), true), Map.entry(new Position(6, 3), true), Map.entry(new Position(6, 4), false), Map.entry(new Position(6, 5), false), Map.entry(new Position(6, 6), true), Map.entry(new Position(6, 7), false), Map.entry(new Position(6, 8), false), Map.entry(new Position(6, 9), false), Map.entry(new Position(6, 10), false), Map.entry(new Position(6, 11), false), Map.entry(new Position(6, 12), false), Map.entry(new Position(6, 13), false), Map.entry(new Position(6, 14), true), Map.entry(new Position(6, 15), false), Map.entry(new Position(7, 0), false), Map.entry(new Position(7, 1), false), Map.entry(new Position(7, 2), true), Map.entry(new Position(7, 3), false), Map.entry(new Position(7, 4), false), Map.entry(new Position(7, 5), false), Map.entry(new Position(7, 6), true), Map.entry(new Position(7, 7), false), Map.entry(new Position(7, 8), false), Map.entry(new Position(7, 9), false), Map.entry(new Position(7, 10), false), Map.entry(new Position(7, 11), false), Map.entry(new Position(7, 12), false), Map.entry(new Position(7, 13), false), Map.entry(new Position(7, 14), true), Map.entry(new Position(7, 15), false), Map.entry(new Position(8, 0), false), Map.entry(new Position(8, 1), true), Map.entry(new Position(8, 2), true), Map.entry(new Position(8, 3), false), Map.entry(new Position(8, 4), false), Map.entry(new Position(8, 5), false), Map.entry(new Position(8, 6), true), Map.entry(new Position(8, 7), false), Map.entry(new Position(8, 8), false), Map.entry(new Position(8, 9), false), Map.entry(new Position(8, 10), false), Map.entry(new Position(8, 11), false), Map.entry(new Position(8, 12), false), Map.entry(new Position(8, 13), false), Map.entry(new Position(8, 14), true), Map.entry(new Position(8, 15), false), Map.entry(new Position(9, 0), false), Map.entry(new Position(9, 1), true), Map.entry(new Position(9, 2), false), Map.entry(new Position(9, 3), false), Map.entry(new Position(9, 4), false), Map.entry(new Position(9, 5), false), Map.entry(new Position(9, 6), true), Map.entry(new Position(9, 7), false), Map.entry(new Position(9, 8), false), Map.entry(new Position(9, 9), false), Map.entry(new Position(9, 10), false), Map.entry(new Position(9, 11), false), Map.entry(new Position(9, 12), false), Map.entry(new Position(9, 13), false), Map.entry(new Position(9, 14), true), Map.entry(new Position(9, 15), false), Map.entry(new Position(10, 0), false), Map.entry(new Position(10, 1), true), Map.entry(new Position(10, 2), false), Map.entry(new Position(10, 3), false), Map.entry(new Position(10, 4), false), Map.entry(new Position(10, 5), false), Map.entry(new Position(10, 6), true), Map.entry(new Position(10, 7), false), Map.entry(new Position(10, 8), false), Map.entry(new Position(10, 9), false), Map.entry(new Position(10, 10), false), Map.entry(new Position(10, 11), false), Map.entry(new Position(10, 12), false), Map.entry(new Position(10, 13), true), Map.entry(new Position(10, 14), true), Map.entry(new Position(10, 15), false), Map.entry(new Position(11, 0), false), Map.entry(new Position(11, 1), false), Map.entry(new Position(11, 2), false), Map.entry(new Position(11, 3), false), Map.entry(new Position(11, 4), false), Map.entry(new Position(11, 5), false), Map.entry(new Position(11, 6), false), Map.entry(new Position(11, 7), true), Map.entry(new Position(11, 8), false), Map.entry(new Position(11, 9), false), Map.entry(new Position(11, 10), false), Map.entry(new Position(11, 11), false), Map.entry(new Position(11, 12), false), Map.entry(new Position(11, 13), true), Map.entry(new Position(11, 14), false), Map.entry(new Position(11, 15), false), Map.entry(new Position(12, 0), false), Map.entry(new Position(12, 1), false), Map.entry(new Position(12, 2), false), Map.entry(new Position(12, 3), false), Map.entry(new Position(12, 4), false), Map.entry(new Position(12, 5), false), Map.entry(new Position(12, 6), false), Map.entry(new Position(12, 7), false), Map.entry(new Position(12, 8), true), Map.entry(new Position(12, 9), false), Map.entry(new Position(12, 10), false), Map.entry(new Position(12, 11), false), Map.entry(new Position(12, 12), true), Map.entry(new Position(12, 13), true), Map.entry(new Position(12, 14), false), Map.entry(new Position(12, 15), false), Map.entry(new Position(13, 0), false), Map.entry(new Position(13, 1), false), Map.entry(new Position(13, 2), false), Map.entry(new Position(13, 3), false), Map.entry(new Position(13, 4), false), Map.entry(new Position(13, 5), false), Map.entry(new Position(13, 6), false), Map.entry(new Position(13, 7), false), Map.entry(new Position(13, 8), true), Map.entry(new Position(13, 9), true), Map.entry(new Position(13, 10), true), Map.entry(new Position(13, 11), true), Map.entry(new Position(13, 12), false), Map.entry(new Position(13, 13), false), Map.entry(new Position(13, 14), false), Map.entry(new Position(13, 15), false), Map.entry(new Position(14, 0), false), Map.entry(new Position(14, 1), false), Map.entry(new Position(14, 2), false), Map.entry(new Position(14, 3), false), Map.entry(new Position(14, 4), false), Map.entry(new Position(14, 5), false), Map.entry(new Position(14, 6), false), Map.entry(new Position(14, 7), false), Map.entry(new Position(14, 8), false), Map.entry(new Position(14, 9), false), Map.entry(new Position(14, 10), false), Map.entry(new Position(14, 11), false), Map.entry(new Position(14, 12), false), Map.entry(new Position(14, 13), false), Map.entry(new Position(14, 14), false), Map.entry(new Position(14, 15), false), Map.entry(new Position(15, 0), false), Map.entry(new Position(15, 1), false), Map.entry(new Position(15, 2), false), Map.entry(new Position(15, 3), false), Map.entry(new Position(15, 4), false), Map.entry(new Position(15, 5), false), Map.entry(new Position(15, 6), false), Map.entry(new Position(15, 7), false), Map.entry(new Position(15, 8), false), Map.entry(new Position(15, 9), false), Map.entry(new Position(15, 10), false), Map.entry(new Position(15, 11), false), Map.entry(new Position(15, 12), false), Map.entry(new Position(15, 13), false), Map.entry(new Position(15, 14), false), Map.entry(new Position(15, 15), false));
        Map<Position, Boolean> mapof9 = Map.<Position, Boolean>ofEntries(Map.entry(new Position(0, 0), false), Map.entry(new Position(0, 1), false), Map.entry(new Position(0, 2), false), Map.entry(new Position(0, 3), false), Map.entry(new Position(0, 4), false), Map.entry(new Position(0, 5), false), Map.entry(new Position(0, 6), false), Map.entry(new Position(0, 7), false), Map.entry(new Position(0, 8), false), Map.entry(new Position(0, 9), false), Map.entry(new Position(0, 10), false), Map.entry(new Position(0, 11), false), Map.entry(new Position(0, 12), false), Map.entry(new Position(0, 13), false), Map.entry(new Position(0, 14), false), Map.entry(new Position(0, 15), false), Map.entry(new Position(1, 0), false), Map.entry(new Position(1, 1), false), Map.entry(new Position(1, 2), false), Map.entry(new Position(1, 3), false), Map.entry(new Position(1, 4), false), Map.entry(new Position(1, 5), false), Map.entry(new Position(1, 6), false), Map.entry(new Position(1, 7), false), Map.entry(new Position(1, 8), false), Map.entry(new Position(1, 9), false), Map.entry(new Position(1, 10), false), Map.entry(new Position(1, 11), false), Map.entry(new Position(1, 12), false), Map.entry(new Position(1, 13), false), Map.entry(new Position(1, 14), false), Map.entry(new Position(1, 15), false), Map.entry(new Position(2, 0), false), Map.entry(new Position(2, 1), false), Map.entry(new Position(2, 2), false), Map.entry(new Position(2, 3), false), Map.entry(new Position(2, 4), false), Map.entry(new Position(2, 5), false), Map.entry(new Position(2, 6), false), Map.entry(new Position(2, 7), false), Map.entry(new Position(2, 8), false), Map.entry(new Position(2, 9), false), Map.entry(new Position(2, 10), false), Map.entry(new Position(2, 11), false), Map.entry(new Position(2, 12), false), Map.entry(new Position(2, 13), false), Map.entry(new Position(2, 14), false), Map.entry(new Position(2, 15), false), Map.entry(new Position(3, 0), false), Map.entry(new Position(3, 1), false), Map.entry(new Position(3, 2), false), Map.entry(new Position(3, 3), false), Map.entry(new Position(3, 4), false), Map.entry(new Position(3, 5), false), Map.entry(new Position(3, 6), false), Map.entry(new Position(3, 7), false), Map.entry(new Position(3, 8), false), Map.entry(new Position(3, 9), false), Map.entry(new Position(3, 10), false), Map.entry(new Position(3, 11), false), Map.entry(new Position(3, 12), false), Map.entry(new Position(3, 13), false), Map.entry(new Position(3, 14), false), Map.entry(new Position(3, 15), false), Map.entry(new Position(4, 0), false), Map.entry(new Position(4, 1), false), Map.entry(new Position(4, 2), false), Map.entry(new Position(4, 3), false), Map.entry(new Position(4, 4), false), Map.entry(new Position(4, 5), false), Map.entry(new Position(4, 6), false), Map.entry(new Position(4, 7), false), Map.entry(new Position(4, 8), false), Map.entry(new Position(4, 9), false), Map.entry(new Position(4, 10), false), Map.entry(new Position(4, 11), true), Map.entry(new Position(4, 12), true), Map.entry(new Position(4, 13), false), Map.entry(new Position(4, 14), false), Map.entry(new Position(4, 15), false), Map.entry(new Position(5, 0), false), Map.entry(new Position(5, 1), false), Map.entry(new Position(5, 2), false), Map.entry(new Position(5, 3), true), Map.entry(new Position(5, 4), true), Map.entry(new Position(5, 5), true), Map.entry(new Position(5, 6), false), Map.entry(new Position(5, 7), false), Map.entry(new Position(5, 8), false), Map.entry(new Position(5, 9), false), Map.entry(new Position(5, 10), false), Map.entry(new Position(5, 11), false), Map.entry(new Position(5, 12), true), Map.entry(new Position(5, 13), false), Map.entry(new Position(5, 14), false), Map.entry(new Position(5, 15), false), Map.entry(new Position(6, 0), false), Map.entry(new Position(6, 1), false), Map.entry(new Position(6, 2), true), Map.entry(new Position(6, 3), true), Map.entry(new Position(6, 4), false), Map.entry(new Position(6, 5), true), Map.entry(new Position(6, 6), false), Map.entry(new Position(6, 7), false), Map.entry(new Position(6, 8), false), Map.entry(new Position(6, 9), false), Map.entry(new Position(6, 10), false), Map.entry(new Position(6, 11), false), Map.entry(new Position(6, 12), true), Map.entry(new Position(6, 13), false), Map.entry(new Position(6, 14), false), Map.entry(new Position(6, 15), false), Map.entry(new Position(7, 0), false), Map.entry(new Position(7, 1), false), Map.entry(new Position(7, 2), true), Map.entry(new Position(7, 3), false), Map.entry(new Position(7, 4), false), Map.entry(new Position(7, 5), true), Map.entry(new Position(7, 6), false), Map.entry(new Position(7, 7), false), Map.entry(new Position(7, 8), false), Map.entry(new Position(7, 9), false), Map.entry(new Position(7, 10), false), Map.entry(new Position(7, 11), false), Map.entry(new Position(7, 12), true), Map.entry(new Position(7, 13), false), Map.entry(new Position(7, 14), false), Map.entry(new Position(7, 15), false), Map.entry(new Position(8, 0), false), Map.entry(new Position(8, 1), false), Map.entry(new Position(8, 2), true), Map.entry(new Position(8, 3), false), Map.entry(new Position(8, 4), false), Map.entry(new Position(8, 5), true), Map.entry(new Position(8, 6), false), Map.entry(new Position(8, 7), false), Map.entry(new Position(8, 8), false), Map.entry(new Position(8, 9), false), Map.entry(new Position(8, 10), false), Map.entry(new Position(8, 11), true), Map.entry(new Position(8, 12), true), Map.entry(new Position(8, 13), false), Map.entry(new Position(8, 14), false), Map.entry(new Position(8, 15), false), Map.entry(new Position(9, 0), false), Map.entry(new Position(9, 1), false), Map.entry(new Position(9, 2), true), Map.entry(new Position(9, 3), true), Map.entry(new Position(9, 4), true), Map.entry(new Position(9, 5), true), Map.entry(new Position(9, 6), true), Map.entry(new Position(9, 7), true), Map.entry(new Position(9, 8), true), Map.entry(new Position(9, 9), true), Map.entry(new Position(9, 10), true), Map.entry(new Position(9, 11), true), Map.entry(new Position(9, 12), false), Map.entry(new Position(9, 13), false), Map.entry(new Position(9, 14), false), Map.entry(new Position(9, 15), false), Map.entry(new Position(10, 0), false), Map.entry(new Position(10, 1), false), Map.entry(new Position(10, 2), false), Map.entry(new Position(10, 3), false), Map.entry(new Position(10, 4), false), Map.entry(new Position(10, 5), false), Map.entry(new Position(10, 6), false), Map.entry(new Position(10, 7), false), Map.entry(new Position(10, 8), false), Map.entry(new Position(10, 9), false), Map.entry(new Position(10, 10), false), Map.entry(new Position(10, 11), false), Map.entry(new Position(10, 12), false), Map.entry(new Position(10, 13), false), Map.entry(new Position(10, 14), false), Map.entry(new Position(10, 15), false), Map.entry(new Position(11, 0), false), Map.entry(new Position(11, 1), false), Map.entry(new Position(11, 2), false), Map.entry(new Position(11, 3), false), Map.entry(new Position(11, 4), false), Map.entry(new Position(11, 5), false), Map.entry(new Position(11, 6), false), Map.entry(new Position(11, 7), false), Map.entry(new Position(11, 8), false), Map.entry(new Position(11, 9), false), Map.entry(new Position(11, 10), false), Map.entry(new Position(11, 11), false), Map.entry(new Position(11, 12), false), Map.entry(new Position(11, 13), false), Map.entry(new Position(11, 14), false), Map.entry(new Position(11, 15), false), Map.entry(new Position(12, 0), false), Map.entry(new Position(12, 1), false), Map.entry(new Position(12, 2), false), Map.entry(new Position(12, 3), false), Map.entry(new Position(12, 4), false), Map.entry(new Position(12, 5), false), Map.entry(new Position(12, 6), false), Map.entry(new Position(12, 7), false), Map.entry(new Position(12, 8), false), Map.entry(new Position(12, 9), false), Map.entry(new Position(12, 10), false), Map.entry(new Position(12, 11), false), Map.entry(new Position(12, 12), false), Map.entry(new Position(12, 13), false), Map.entry(new Position(12, 14), false), Map.entry(new Position(12, 15), false), Map.entry(new Position(13, 0), false), Map.entry(new Position(13, 1), false), Map.entry(new Position(13, 2), false), Map.entry(new Position(13, 3), false), Map.entry(new Position(13, 4), false), Map.entry(new Position(13, 5), false), Map.entry(new Position(13, 6), false), Map.entry(new Position(13, 7), false), Map.entry(new Position(13, 8), false), Map.entry(new Position(13, 9), false), Map.entry(new Position(13, 10), false), Map.entry(new Position(13, 11), false), Map.entry(new Position(13, 12), false), Map.entry(new Position(13, 13), false), Map.entry(new Position(13, 14), false), Map.entry(new Position(13, 15), false), Map.entry(new Position(14, 0), false), Map.entry(new Position(14, 1), false), Map.entry(new Position(14, 2), false), Map.entry(new Position(14, 3), false), Map.entry(new Position(14, 4), false), Map.entry(new Position(14, 5), false), Map.entry(new Position(14, 6), false), Map.entry(new Position(14, 7), false), Map.entry(new Position(14, 8), false), Map.entry(new Position(14, 9), false), Map.entry(new Position(14, 10), false), Map.entry(new Position(14, 11), false), Map.entry(new Position(14, 12), false), Map.entry(new Position(14, 13), false), Map.entry(new Position(14, 14), false), Map.entry(new Position(14, 15), false), Map.entry(new Position(15, 0), false), Map.entry(new Position(15, 1), false), Map.entry(new Position(15, 2), false), Map.entry(new Position(15, 3), false), Map.entry(new Position(15, 4), false), Map.entry(new Position(15, 5), false), Map.entry(new Position(15, 6), false), Map.entry(new Position(15, 7), false), Map.entry(new Position(15, 8), false), Map.entry(new Position(15, 9), false), Map.entry(new Position(15, 10), false), Map.entry(new Position(15, 11), false), Map.entry(new Position(15, 12), false), Map.entry(new Position(15, 13), false), Map.entry(new Position(15, 14), false), Map.entry(new Position(15, 15), false));
        Map<Position, Boolean> mapof5 = Map.<Position, Boolean>ofEntries(Map.entry(new Position(0, 0), false), Map.entry(new Position(0, 1), false), Map.entry(new Position(0, 2), false), Map.entry(new Position(0, 3), false), Map.entry(new Position(0, 4), false), Map.entry(new Position(0, 5), false), Map.entry(new Position(0, 6), false), Map.entry(new Position(0, 7), false), Map.entry(new Position(0, 8), false), Map.entry(new Position(0, 9), false), Map.entry(new Position(0, 10), false), Map.entry(new Position(0, 11), false), Map.entry(new Position(0, 12), false), Map.entry(new Position(0, 13), false), Map.entry(new Position(0, 14), false), Map.entry(new Position(0, 15), false), Map.entry(new Position(1, 0), false), Map.entry(new Position(1, 1), false), Map.entry(new Position(1, 2), false), Map.entry(new Position(1, 3), false), Map.entry(new Position(1, 4), false), Map.entry(new Position(1, 5), false), Map.entry(new Position(1, 6), false), Map.entry(new Position(1, 7), false), Map.entry(new Position(1, 8), false), Map.entry(new Position(1, 9), false), Map.entry(new Position(1, 10), false), Map.entry(new Position(1, 11), false), Map.entry(new Position(1, 12), false), Map.entry(new Position(1, 13), false), Map.entry(new Position(1, 14), false), Map.entry(new Position(1, 15), false), Map.entry(new Position(2, 0), false), Map.entry(new Position(2, 1), false), Map.entry(new Position(2, 2), false), Map.entry(new Position(2, 3), false), Map.entry(new Position(2, 4), false), Map.entry(new Position(2, 5), false), Map.entry(new Position(2, 6), false), Map.entry(new Position(2, 7), false), Map.entry(new Position(2, 8), false), Map.entry(new Position(2, 9), false), Map.entry(new Position(2, 10), false), Map.entry(new Position(2, 11), false), Map.entry(new Position(2, 12), false), Map.entry(new Position(2, 13), false), Map.entry(new Position(2, 14), false), Map.entry(new Position(2, 15), false), Map.entry(new Position(3, 0), false), Map.entry(new Position(3, 1), false), Map.entry(new Position(3, 2), false), Map.entry(new Position(3, 3), false), Map.entry(new Position(3, 4), false), Map.entry(new Position(3, 5), false), Map.entry(new Position(3, 6), false), Map.entry(new Position(3, 7), false), Map.entry(new Position(3, 8), false), Map.entry(new Position(3, 9), false), Map.entry(new Position(3, 10), false), Map.entry(new Position(3, 11), false), Map.entry(new Position(3, 12), false), Map.entry(new Position(3, 13), false), Map.entry(new Position(3, 14), false), Map.entry(new Position(3, 15), false), Map.entry(new Position(4, 0), false), Map.entry(new Position(4, 1), false), Map.entry(new Position(4, 2), false), Map.entry(new Position(4, 3), false), Map.entry(new Position(4, 4), false), Map.entry(new Position(4, 5), false), Map.entry(new Position(4, 6), false), Map.entry(new Position(4, 7), false), Map.entry(new Position(4, 8), false), Map.entry(new Position(4, 9), false), Map.entry(new Position(4, 10), false), Map.entry(new Position(4, 11), false), Map.entry(new Position(4, 12), false), Map.entry(new Position(4, 13), false), Map.entry(new Position(4, 14), false), Map.entry(new Position(4, 15), false), Map.entry(new Position(5, 0), false), Map.entry(new Position(5, 1), false), Map.entry(new Position(5, 2), true), Map.entry(new Position(5, 3), true), Map.entry(new Position(5, 4), true), Map.entry(new Position(5, 5), true), Map.entry(new Position(5, 6), true), Map.entry(new Position(5, 7), true), Map.entry(new Position(5, 8), false), Map.entry(new Position(5, 9), false), Map.entry(new Position(5, 10), false), Map.entry(new Position(5, 11), false), Map.entry(new Position(5, 12), true), Map.entry(new Position(5, 13), true), Map.entry(new Position(5, 14), false), Map.entry(new Position(5, 15), false), Map.entry(new Position(6, 0), false), Map.entry(new Position(6, 1), false), Map.entry(new Position(6, 2), true), Map.entry(new Position(6, 3), false), Map.entry(new Position(6, 4), false), Map.entry(new Position(6, 5), false), Map.entry(new Position(6, 6), false), Map.entry(new Position(6, 7), true), Map.entry(new Position(6, 8), false), Map.entry(new Position(6, 9), false), Map.entry(new Position(6, 10), false), Map.entry(new Position(6, 11), false), Map.entry(new Position(6, 12), false), Map.entry(new Position(6, 13), true), Map.entry(new Position(6, 14), false), Map.entry(new Position(6, 15), false), Map.entry(new Position(7, 0), false), Map.entry(new Position(7, 1), false), Map.entry(new Position(7, 2), true), Map.entry(new Position(7, 3), false), Map.entry(new Position(7, 4), false), Map.entry(new Position(7, 5), false), Map.entry(new Position(7, 6), false), Map.entry(new Position(7, 7), true), Map.entry(new Position(7, 8), false), Map.entry(new Position(7, 9), false), Map.entry(new Position(7, 10), false), Map.entry(new Position(7, 11), false), Map.entry(new Position(7, 12), true), Map.entry(new Position(7, 13), true), Map.entry(new Position(7, 14), false), Map.entry(new Position(7, 15), false), Map.entry(new Position(8, 0), false), Map.entry(new Position(8, 1), false), Map.entry(new Position(8, 2), true), Map.entry(new Position(8, 3), false), Map.entry(new Position(8, 4), false), Map.entry(new Position(8, 5), false), Map.entry(new Position(8, 6), false), Map.entry(new Position(8, 7), true), Map.entry(new Position(8, 8), false), Map.entry(new Position(8, 9), false), Map.entry(new Position(8, 10), false), Map.entry(new Position(8, 11), false), Map.entry(new Position(8, 12), true), Map.entry(new Position(8, 13), false), Map.entry(new Position(8, 14), false), Map.entry(new Position(8, 15), false), Map.entry(new Position(9, 0), false), Map.entry(new Position(9, 1), false), Map.entry(new Position(9, 2), true), Map.entry(new Position(9, 3), false), Map.entry(new Position(9, 4), false), Map.entry(new Position(9, 5), false), Map.entry(new Position(9, 6), false), Map.entry(new Position(9, 7), true), Map.entry(new Position(9, 8), true), Map.entry(new Position(9, 9), false), Map.entry(new Position(9, 10), true), Map.entry(new Position(9, 11), true), Map.entry(new Position(9, 12), true), Map.entry(new Position(9, 13), false), Map.entry(new Position(9, 14), false), Map.entry(new Position(9, 15), false), Map.entry(new Position(10, 0), false), Map.entry(new Position(10, 1), false), Map.entry(new Position(10, 2), false), Map.entry(new Position(10, 3), false), Map.entry(new Position(10, 4), false), Map.entry(new Position(10, 5), false), Map.entry(new Position(10, 6), false), Map.entry(new Position(10, 7), false), Map.entry(new Position(10, 8), true), Map.entry(new Position(10, 9), true), Map.entry(new Position(10, 10), true), Map.entry(new Position(10, 11), false), Map.entry(new Position(10, 12), false), Map.entry(new Position(10, 13), false), Map.entry(new Position(10, 14), false), Map.entry(new Position(10, 15), false), Map.entry(new Position(11, 0), false), Map.entry(new Position(11, 1), false), Map.entry(new Position(11, 2), false), Map.entry(new Position(11, 3), false), Map.entry(new Position(11, 4), false), Map.entry(new Position(11, 5), false), Map.entry(new Position(11, 6), false), Map.entry(new Position(11, 7), false), Map.entry(new Position(11, 8), false), Map.entry(new Position(11, 9), false), Map.entry(new Position(11, 10), false), Map.entry(new Position(11, 11), false), Map.entry(new Position(11, 12), false), Map.entry(new Position(11, 13), false), Map.entry(new Position(11, 14), false), Map.entry(new Position(11, 15), false), Map.entry(new Position(12, 0), false), Map.entry(new Position(12, 1), false), Map.entry(new Position(12, 2), false), Map.entry(new Position(12, 3), false), Map.entry(new Position(12, 4), false), Map.entry(new Position(12, 5), false), Map.entry(new Position(12, 6), false), Map.entry(new Position(12, 7), false), Map.entry(new Position(12, 8), false), Map.entry(new Position(12, 9), false), Map.entry(new Position(12, 10), false), Map.entry(new Position(12, 11), false), Map.entry(new Position(12, 12), false), Map.entry(new Position(12, 13), false), Map.entry(new Position(12, 14), false), Map.entry(new Position(12, 15), false), Map.entry(new Position(13, 0), false), Map.entry(new Position(13, 1), false), Map.entry(new Position(13, 2), false), Map.entry(new Position(13, 3), false), Map.entry(new Position(13, 4), false), Map.entry(new Position(13, 5), false), Map.entry(new Position(13, 6), false), Map.entry(new Position(13, 7), false), Map.entry(new Position(13, 8), false), Map.entry(new Position(13, 9), false), Map.entry(new Position(13, 10), false), Map.entry(new Position(13, 11), false), Map.entry(new Position(13, 12), false), Map.entry(new Position(13, 13), false), Map.entry(new Position(13, 14), false), Map.entry(new Position(13, 15), false), Map.entry(new Position(14, 0), false), Map.entry(new Position(14, 1), false), Map.entry(new Position(14, 2), false), Map.entry(new Position(14, 3), false), Map.entry(new Position(14, 4), false), Map.entry(new Position(14, 5), false), Map.entry(new Position(14, 6), false), Map.entry(new Position(14, 7), false), Map.entry(new Position(14, 8), false), Map.entry(new Position(14, 9), false), Map.entry(new Position(14, 10), false), Map.entry(new Position(14, 11), false), Map.entry(new Position(14, 12), false), Map.entry(new Position(14, 13), false), Map.entry(new Position(14, 14), false), Map.entry(new Position(14, 15), false), Map.entry(new Position(15, 0), false), Map.entry(new Position(15, 1), false), Map.entry(new Position(15, 2), false), Map.entry(new Position(15, 3), false), Map.entry(new Position(15, 4), false), Map.entry(new Position(15, 5), false), Map.entry(new Position(15, 6), false), Map.entry(new Position(15, 7), false), Map.entry(new Position(15, 8), false), Map.entry(new Position(15, 9), false), Map.entry(new Position(15, 10), false), Map.entry(new Position(15, 11), false), Map.entry(new Position(15, 12), false), Map.entry(new Position(15, 13), false), Map.entry(new Position(15, 14), false), Map.entry(new Position(15, 15), false));

        DigitRecognitionModel model = new DigitRecognitionModel();
        model.train();

        DigitRecognitionModel.DRAWING = mapof6;
        System.out.println(model.getOutput());

        JSONObject model0 = new JSONObject();
        int index = 0;
        for (NeuralLayer layer : model.network.layers) {
            JSONObject layer0 = new JSONObject();

            int i = 0;
            for (Neuron neuron : layer.neurons) {
                i++;
                JSONObject neuron0 = new JSONObject();

                neuron0.put("bias", neuron.bias);
                int i1 = 0;
                for (BigDecimal value : neuron.connectionWeights.values()) {
                    i1++;
                    neuron0.put("weights" + i1, value.doubleValue());
                }

                layer0.put("neuron" + i, neuron0);
            }

            model0.put("layer" + ++index, layer0);
        }

        File f = new File("model.json");
        FileOutputStream fos = new FileOutputStream(f);
        fos.write(model0.toString().getBytes());
        fos.close();
    }
}