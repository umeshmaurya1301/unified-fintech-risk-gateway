package com.ufrg.grader;

import java.util.Map;

/**
 * GraderFactory — Java equivalent of Python's {@code get_grader(task_name)}.
 *
 * <p>Usage:
 * <pre>
 *   TaskGrader grader = GraderFactory.getGrader("hard");
 *   double score = grader.grade(trajectory);
 * </pre>
 */
public final class GraderFactory {

    private static final Map<String, Class<? extends TaskGrader>> GRADER_MAP = Map.of(
            "easy",   EasyGrader.class,
            "medium", MediumGrader.class,
            "hard",   HardGrader.class
    );

    private GraderFactory() { /* utility class */ }

    /**
     * Return the appropriate {@link TaskGrader} instance for the given task name.
     *
     * @param taskName one of {@code "easy"}, {@code "medium"}, or {@code "hard"}
     * @return a fresh grader instance
     * @throws IllegalArgumentException if {@code taskName} is unrecognised
     */
    public static TaskGrader getGrader(String taskName) {
        Class<? extends TaskGrader> clazz = GRADER_MAP.get(taskName);
        if (clazz == null) {
            throw new IllegalArgumentException(
                    "Unknown task '" + taskName + "'. Expected one of: " + GRADER_MAP.keySet()
            );
        }
        try {
            return clazz.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to instantiate grader for task: " + taskName, e);
        }
    }
}
