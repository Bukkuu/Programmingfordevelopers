
// Question no. 1(a)

// public class MinimumCostClothing {
//     public static int minCost(int[][] price) {
//         int n = price.length;

//         int[][] dp = new int[n][3];
//         dp[0][0] = price[0][0];
//         dp[0][1] = price[0][1];
//         dp[0][2] = price[0][2];

//         for (int i = 1; i < n; i++) {
//             dp[i][0] = price[i][0] + Math.min(dp[i - 1][1], dp[i - 1][2]);
//             dp[i][1] = price[i][1] + Math.min(dp[i - 1][0], dp[i - 1][2]);
//             dp[i][2] = price[i][2] + Math.min(dp[i - 1][0], dp[i - 1][1]);
//         }

//         return Math.min(dp[n - 1][0], Math.min(dp[n - 1][1], dp[n - 1][2]));
//     }

//     public static void main(String[] args) {
//         int[][] price = {
//             {14, 4, 11},
//             {11, 14, 3},
//             {14, 2, 10}
//         };
//         int result = minCost(price);
//         System.out.println("Minimum cost: " + result);
//     }
// }

// Q no. 1(b)

// public class MinimumCoinsDistribution {
//     public static int minCoins(int[] ratings) {
//         int n = ratings.length;
//         int[] coins = new int[n];

//         // Initialize each rider with 1 coin
//         for (int i = 0; i < n; i++) {
//             coins[i] = 1;
//         }

//         // Traverse from left to right and ensure higher-rated riders get more coins
//         for (int i = 1; i < n; i++) {
//             if (ratings[i] > ratings[i - 1]) {
//                 coins[i] = coins[i - 1] + 1;
//             }
//         }

//         // Traverse from right to left to handle cases where higher-rated riders are on the right
//         for (int i = n - 2; i >= 0; i--) {
//             if (ratings[i] > ratings[i + 1]) {
//                 coins[i] = Math.max(coins[i], coins[i + 1] + 1);
//             }
//         }

//         // Calculate the total number of coins required
//         int totalCoins = 0;
//         for (int coin : coins) {
//             totalCoins += coin;
//         }

//         return totalCoins;
//     }

//     public static void main(String[] args) {
//         int[] ratings = {1, 0, 2};
//         int result = minCoins(ratings);
//         System.out.println("Minimum coins required: " + result);
//     }
// }

// Q no.2(a)

// public class LongestDecreasingSubsequence {
//     public static int longestSubsequence(int[] nums, int k) {
//         int n = nums.length;
//         int[] dp = new int[n];
//         int maxLen = 1; // Minimum length of the subsequence is 1

//         for (int i = 0; i < n; i++) {
//             dp[i] = 1; // Initialize the length of subsequence ending at index i to 1
//             for (int j = 0; j < i; j++) {
//                 // Check if the conditions are satisfied to extend the subsequence
//                 if (nums[i] < nums[j] && Math.abs(nums[i] - nums[j]) <= k) {
//                     dp[i] = Math.max(dp[i], dp[j] + 1);
//                 }
//             }
//             maxLen = Math.max(maxLen, dp[i]);
//         }

//         return maxLen;
//     }

//     public static void main(String[] args) {
//         int[] nums = {8, 5, 4, 2, 1, 4, 3, 4, 3, 1, 15};
//         int k = 3;
//         int result = longestSubsequence(nums, k);
//         System.out.println("Longest decreasing subsequence length: " + result);
//     }
// }

// Q no. 2(b)
// import java.util.*;

// class WhitelistRandom {
//     private Random random;
//     private List<Integer> whitelist;
//     private Map<Integer, Integer> portCount;

//     public WhitelistRandom(int k, int[] blacklistedPorts) {
//         random = new Random();
//         whitelist = new ArrayList<>();
//         portCount = new HashMap<>();

//         for (int i = 0; i < k; i++) {
//             portCount.put(i, 1);
//             whitelist.add(i);
//         }

//         for (int port : blacklistedPorts) {
//             if (portCount.containsKey(port)) {
//                 portCount.remove(port);
//                 whitelist.remove(Integer.valueOf(port));
//             }
//         }
//     }

//     public int get() {
//         int idx = random.nextInt(whitelist.size());
//         int port = whitelist.get(idx);

//         int count = portCount.get(port);
//         if (count > 1) {
//             portCount.put(port, count - 1);
//         } else {
//             portCount.remove(port);
//             whitelist.remove(idx);
//         }

//         return port;
//     }
// }


// public class Main {
//     public static void main(String[] args) {
//         WhitelistRandom p = new WhitelistRandom(7, new int[]{2, 3, 5});
//         System.out.println(p.get()); // Return a whitelisted random port
//         System.out.println(p.get()); // Return another whitelisted random port
//         System.out.println(p.get()); // Return another whitelisted random port
//     }
// }

// Q no. 3(a)


// public class MaxPointsFromTargets {
//     public static int maxPoints(int[] a) {
//         int n = a.length;
//         int[] targets = new int[n + 2];
//         targets[0] = targets[n + 1] = 1; // Padding with 1s at the beginning and end
//         System.arraycopy(a, 0, targets, 1, n);

//         int[][] dp = new int[n + 2][n + 2];

//         for (int len = 1; len <= n; len++) {
//             for (int i = 1; i + len - 1 <= n; i++) {
//                 int j = i + len - 1;
//                 for (int k = i; k <= j; k++) {
//                     dp[i][j] = Math.max(dp[i][j], dp[i][k - 1] + dp[k + 1][j] + targets[i - 1] * targets[k] * targets[j + 1]);
//                 }
//             }
//         }

//         return dp[1][n];
//     }

//     public static void main(String[] args) {
//         int[] a = {3, 1, 5, 8};
//         int result = maxPoints(a);
//         System.out.println("Maximum points: " + result);
//     }
// }

// Q no. 3(b)

// Bellman-Ford Algorithm:

// import java.util.Arrays;

// class BellmanFord {
//     static class Edge {
//         int source, destination, weight;

//         Edge(int source, int destination, int weight) {
//             this.source = source;
//             this.destination = destination;
//             this.weight = weight;
//         }
//     }

//     static int[] bellmanFord(int vertices, Edge[] edges, int source) {
//         int[] distances = new int[vertices];
//         Arrays.fill(distances, Integer.MAX_VALUE);
//         distances[source] = 0;

//         for (int i = 0; i < vertices - 1; i++) {
//             for (Edge edge : edges) {
//                 if (distances[edge.source] != Integer.MAX_VALUE &&
//                         distances[edge.source] + edge.weight < distances[edge.destination]) {
//                     distances[edge.destination] = distances[edge.source] + edge.weight;
//                 }
//             }
//         }

//         return distances;
//     }

//     public static void main(String[] args) {
//         int vertices = 5;
//         Edge[] edges = {
//             new Edge(0, 1, -1),
//             new Edge(0, 2, 4),
//             new Edge(1, 2, 3),
//             new Edge(1, 3, 2),
//             new Edge(1, 4, 2),
//             new Edge(3, 2, 5),
//             new Edge(3, 1, 1),
//             new Edge(4, 3, -3)
//         };

//         int source = 0;
//         int[] distances = bellmanFord(vertices, edges, source);

//         System.out.println("Shortest distances from source " + source + ":");
//         for (int i = 0; i < vertices; i++) {
//             System.out.println("Vertex " + i + ": " + distances[i]);
//         }
//     }
// }


// Priority Queue Using Maximum Heap:

// import java.util.PriorityQueue;
// import java.util.Comparator;

// public class MaxHeapPriorityQueue {
//     public static void main(String[] args) {
//         // Create a max-heap priority queue
//         PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());

//         // Add elements to the priority queue
//         maxHeap.add(5);
//         maxHeap.add(10);
//         maxHeap.add(2);
//         maxHeap.add(8);
//         maxHeap.add(3);

//         // Remove and print elements from the priority queue
//         while (!maxHeap.isEmpty()) {
//             System.out.println(maxHeap.poll());
//         }
//     }
// }

// Q no.4(a)

// import java.util.*;

// public class MinimumStepsToCompleteTasks {
//     public static int minSteps(int N, int[][] prerequisites) {
//         List<List<Integer>> graph = new ArrayList<>();
//         for (int i = 0; i <= N; i++) {
//             graph.add(new ArrayList<>());
//         }

//         for (int[] prerequisite : prerequisites) {
//             int x = prerequisite[0];
//             int y = prerequisite[1];
//             graph.get(x).add(y);
//         }

//         int[] depths = new int[N + 1];
//         Arrays.fill(depths, -1);

//         for (int i = 1; i <= N; i++) {
//             if (depths[i] == -1) {
//                 if (!dfs(i, graph, depths)) {
//                     return -1; // Cycle detected, can't complete all tasks
//                 }
//             }
//         }

//         int minSteps = 0;
//         for (int depth : depths) {
//             minSteps = Math.max(minSteps, depth);
//         }

//         return minSteps;
//     }

//     private static boolean dfs(int node, List<List<Integer>> graph, int[] depths) {
//         if (depths[node] != -1) {
//             return true; // Already visited, no cycle detected
//         }

//         depths[node] = 0; // Start exploring the node

//         for (int neighbor : graph.get(node)) {
//             if (!dfs(neighbor, graph, depths)) {
//                 return false; // Cycle detected
//             }
//             depths[node] = Math.max(depths[node], depths[neighbor] + 1);
//         }

//         return true;
//     }

//     public static void main(String[] args) {
//         int N = 3;
//         int[][] prerequisites = {{1, 3}, {2, 3}};
//         int result = minSteps(N, prerequisites);
//         System.out.println("Minimum steps needed: " + result);
//     }
// }


// Q no. 4(b)

// class TreeNode {
//     int val;
//     TreeNode left;
//     TreeNode right;
    
//     TreeNode(int val) {
//         this.val = val;
//         this.left = null;
//         this.right = null;
//     }
// }

// public class BrothersInBinaryTree {
//     private TreeNode parentX = null;
//     private TreeNode parentY = null;
//     private int depthX = -1;
//     private int depthY = -1;

//     public boolean areBrothers(TreeNode root, int x, int y) {
//         dfs(root, null, x, y, 0);
//         return depthX == depthY && parentX != parentY;
//     }

//     private void dfs(TreeNode node, TreeNode parent, int x, int y, int depth) {
//         if (node == null) {
//             return;
//         }
        
//         if (node.val == x) {
//             parentX = parent;
//             depthX = depth;
//         } else if (node.val == y) {
//             parentY = parent;
//             depthY = depth;
//         }
        
//         dfs(node.left, node, x, y, depth + 1);
//         dfs(node.right, node, x, y, depth + 1);
//     }

//     public static void main(String[] args) {
//         TreeNode root = new TreeNode(1);
//         root.left = new TreeNode(2);
//         root.right = new TreeNode(3);
//         root.left.left = new TreeNode(4);
        
//         BrothersInBinaryTree solution = new BrothersInBinaryTree();
//         boolean result = solution.areBrothers(root, 4, 3);
//         System.out.println("Nodes are brothers: " + result);
//     }
// }


// Q no.5(a)

// import java.util.Random;

// public class HillClimbing {
//     static final int MAX_ITERATIONS = 1000;

//     public static double objectiveFunction(double x) {
//         // Replace this with your own objective function
//         return -x * x + 4 * x - 5; // Example: maximizing a quadratic function
//     }

//     public static double hillClimbing(double start, double stepSize) {
//         double current = start;
//         double currentEval = objectiveFunction(current);

//         for (int i = 0; i < MAX_ITERATIONS; i++) {
//             double next = current + stepSize;
//             double nextEval = objectiveFunction(next);

//             if (nextEval > currentEval) {
//                 current = next;
//                 currentEval = nextEval;
//             } else {
//                 stepSize *= -0.5; // Reduce the step size to refine the search
//             }
//         }

//         return current;
//     }

//     public static void main(String[] args) {
//         Random random = new Random();
//         double initialGuess = random.nextDouble() * 10; // Initial guess between 0 and 10
//         double stepSize = 0.1;

//         double bestSolution = hillClimbing(initialGuess, stepSize);
//         System.out.println("Best solution: " + bestSolution);
//         System.out.println("Objective function value: " + objectiveFunction(bestSolution));
//     }
// }

//Q no. 5 (b)

// import java.util.*;

// public class ReorientConnections {
//     public int minReorder(int n, int[][] connections) {
//         List<List<Integer>> graph = new ArrayList<>();
//         for (int i = 0; i < n; i++) {
//             graph.add(new ArrayList<>());
//         }
        
//         for (int[] conn : connections) {
//             graph.get(conn[0]).add(conn[1]); // Original direction
//             graph.get(conn[1]).add(-conn[0]); // Reversed direction
//         }
        
//         return dfs(0, graph, new boolean[n]);
//     }
    
//     private int dfs(int node, List<List<Integer>> graph, boolean[] visited) {
//         visited[node] = true;
//         int changes = 0;
        
//         for (int neighbor : graph.get(node)) {
//             if (!visited[Math.abs(neighbor)]) {
//                 changes += dfs(Math.abs(neighbor), graph, visited) + (neighbor > 0 ? 1 : 0);
//             }
//         }
        
//         return changes;
//     }

//     public static void main(String[] args) {
//         ReorientConnections solution = new ReorientConnections();
//         int n = 6;
//         int[][] connections = {{0, 1}, {1, 3}, {2, 3}, {4, 0}, {4, 5}};
//         int result = solution.minReorder(n, connections);
//         System.out.println("Minimum changes required: " + result);
//     }
// }


// Q No. 6

// import java.util.Arrays;

// public class ParallelMergeSort {
//     private static final int THREAD_THRESHOLD = 1000;

//     public static void main(String[] args) {
//         int[] arr = {5, 2, 9, 1, 5, 6, 3, 7, 8};
        
//         parallelMergeSort(arr, 0, arr.length - 1);
        
//         System.out.println(Arrays.toString(arr));
//     }

//     public static void parallelMergeSort(int[] arr, int left, int right) {
//         if (left < right) {
//             int mid = left + (right - left) / 2;
            
//             if (right - left < THREAD_THRESHOLD) {
//                 // Sort the subarray using a single thread
//                 mergeSort(arr, left, right);
//             } else {
//                 // Sort the left and right subarrays using separate threads
//                 Thread leftThread = new Thread(() -> parallelMergeSort(arr, left, mid));
//                 Thread rightThread = new Thread(() -> parallelMergeSort(arr, mid + 1, right));
                
//                 leftThread.start();
//                 rightThread.start();
                
//                 try {
//                     leftThread.join();
//                     rightThread.join();
//                 } catch (InterruptedException e) {
//                     e.printStackTrace();
//                 }
                
//                 merge(arr, left, mid, right);
//             }
//         }
//     }

//     public static void mergeSort(int[] arr, int left, int right) {
//         if (left < right) {
//             int mid = left + (right - left) / 2;
//             mergeSort(arr, left, mid);
//             mergeSort(arr, mid + 1, right);
//             merge(arr, left, mid, right);
//         }
//     }

//     public static void merge(int[] arr, int left, int mid, int right) {
//         int[] temp = new int[right - left + 1];
//         int i = left, j = mid + 1, k = 0;
        
//         while (i <= mid && j <= right) {
//             if (arr[i] <= arr[j]) {
//                 temp[k++] = arr[i++];
//             } else {
//                 temp[k++] = arr[j++];
//             }
//         }
        
//         while (i <= mid) {
//             temp[k++] = arr[i++];
//         }
        
//         while (j <= right) {
//             temp[k++] = arr[j++];
//         }
        
//         for (int p = 0; p < temp.length; p++) {
//             arr[left + p] = temp[p];
//         }
//     }
// }


// Question no. 7

// import javafx.application.Application;
// import javafx.scene.Scene;
// import javafx.scene.canvas.Canvas;
// import javafx.scene.control.ToolBar;
// import javafx.scene.layout.BorderPane;
// import javafx.stage.Stage;

// public class GraphApp extends Application {

//     private Canvas canvas;

//     public static void main(String[] args) {
//         launch(args);
//     }

//     @Override
//     public void start(Stage primaryStage) {
//         BorderPane root = new BorderPane();
//         canvas = new Canvas(800, 600);
//         root.setCenter(canvas);
//         ToolBar toolBar = createToolBar();
//         root.setTop(toolBar);

//         Scene scene = new Scene(root, 800, 600);
//         primaryStage.setScene(scene);
//         primaryStage.setTitle("Social Network Graph");
//         primaryStage.show();
//     }

//     private ToolBar createToolBar() {
//         // Create and configure toolbar buttons
//         // Add event listeners for button actions

//         ToolBar toolBar = new ToolBar(/* Add your buttons here */);
//         return toolBar;
//     }

//     // Other methods for drawing graph, handling events, etc.

//     public static void main(String[] args) {
//         launch(args);
//     }
// }
