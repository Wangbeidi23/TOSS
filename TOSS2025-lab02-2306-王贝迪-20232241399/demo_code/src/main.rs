use eframe::egui;
use egui::{
    epaint::{CircleShape, Color32, Shape, Stroke, TextShape}, // Import TextShape
    text::LayoutJob,
    FontId, Pos2, Vec2, // Removed Align2
};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    f32::INFINITY,
};

// --- Constants ---
const NODE_RADIUS: f32 = 20.0;
const INTERACTION_RADIUS: f32 = 25.0; // Slightly larger for easier clicking
const EDGE_INTERACTION_TOLERANCE: f32 = 8.0;

const COLOR_NODE_DEFAULT: Color32 = Color32::from_rgb(60, 150, 255); // Blue
const COLOR_NODE_START: Color32 = Color32::from_rgb(255, 200, 0); // Yellow
const COLOR_NODE_VISITED: Color32 = Color32::from_rgb(255, 50, 50); // Red
const COLOR_EDGE_DEFAULT: Color32 = Color32::from_gray(100);
const COLOR_EDGE_VISITED: Color32 = Color32::GREEN;
const COLOR_TEXT: Color32 = Color32::WHITE;
const COLOR_TEXT_DARK: Color32 = Color32::BLACK;

// --- Data Structures ---

#[derive(Clone)]
struct Node {
    id: usize,
    position: Pos2,
    edges: Vec<Edge>,
}

#[derive(Clone, Copy)]
struct Edge {
    to: usize,
    weight: f32,
    /// Used by BFS/DFS to mark edges traversed during the algorithm run
    visited: bool,
    /// Direction vector for offsetting weight label
    direction: Vec2,
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum Algorithm {
    BFS,
    DFS,
    Dijkstra,
}

struct DijkstraState {
    distances: Vec<f32>,
    heap: BinaryHeap<DijkstraNode>,
    predecessors: HashMap<usize, usize>, // Not currently used for visualization, but useful
    processed_nodes: HashSet<usize>, // Nodes whose final distance is known
}

#[derive(Copy, Clone, PartialEq)]
struct DijkstraNode {
    id: usize,
    distance: f32,
}

impl Eq for DijkstraNode {}
impl PartialOrd for DijkstraNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // BinaryHeap is a max-heap, so we negate the distance to get min-heap behavior
        other.distance.partial_cmp(&self.distance)
    }
}
impl Ord for DijkstraNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Represents the current state of user interaction for creating graph elements.
#[derive(PartialEq)]
enum CreationState {
    /// No creation action is in progress.
    None,
    /// User is about to create a node at the given position (modal window is open).
    CreatingNode(Pos2),
    /// User is dragging to create an edge from the node with the given ID.
    CreatingEdge(usize),
}

/// Represents the current state of user interaction for editing graph elements.
#[derive(PartialEq)]
enum EditingState {
    /// No editing action is in progress.
    None,
    /// User is editing the weight of the edge from 'from' to 'to' (modal window is open).
    EditingEdgeWeight(usize, usize),
}


// --- App State ---

struct GraphApp {
    nodes: Vec<Node>,
    selected_algorithm: Algorithm,

    // Algorithm State
    work_queue_stack: VecDeque<usize>, // Used for BFS (queue) and DFS (stack)
    visited_nodes: HashSet<usize>,      // Used for BFS/DFS to prevent cycles
    dijkstra_state: Option<DijkstraState>,

    // Interaction State
    dragging_node: Option<usize>,
    selected_starts: HashSet<usize>, // Start nodes for algorithms
    creation_state: CreationState,
    editing_state: EditingState,
    next_node_id: usize, // Counter for unique node IDs

    // Visualization Settings
    show_distances: bool,
    step_speed: f32, // Seconds per step
    last_step_time: f64, // Time when the last step occurred
}

// --- Default Implementation ---

impl Default for GraphApp {
    fn default() -> Self {
        let mut nodes = Vec::new();
        // Add some initial nodes
        nodes.push(Node {
            id: 0,
            position: Pos2::new(150.0, 150.0),
            edges: vec![],
        });
        nodes.push(Node {
            id: 1,
            position: Pos2::new(400.0, 150.0),
            edges: vec![],
        });
        nodes.push(Node {
            id: 2,
            position: Pos2::new(275.0, 350.0),
            edges: vec![],
        });

        // Add some initial edges (demonstration purposes)
        // Calculate directions based on positions
        let dir_0_1 = (nodes[1].position - nodes[0].position).normalized().rot90();
        let dir_1_0 = (nodes[0].position - nodes[1].position).normalized().rot90();
        let dir_0_2 = (nodes[2].position - nodes[0].position).normalized().rot90();
        let dir_2_0 = (nodes[0].position - nodes[2].position).normalized().rot90();
        let dir_1_2 = (nodes[2].position - nodes[1].position).normalized().rot90();
        let dir_2_1 = (nodes[1].position - nodes[2].position).normalized().rot90();

        nodes[0].edges.push(Edge { to: 1, weight: 1.0, visited: false, direction: dir_0_1 });
        nodes[1].edges.push(Edge { to: 0, weight: 1.0, visited: false, direction: dir_1_0 });
        nodes[0].edges.push(Edge { to: 2, weight: 3.0, visited: false, direction: dir_0_2 });
        nodes[2].edges.push(Edge { to: 0, weight: 3.0, visited: false, direction: dir_2_0 });
        nodes[1].edges.push(Edge { to: 2, weight: 2.0, visited: false, direction: dir_1_2 });
        nodes[2].edges.push(Edge { to: 1, weight: 2.0, visited: false, direction: dir_2_1 });


        Self {
            nodes,
            selected_algorithm: Algorithm::BFS,
            work_queue_stack: VecDeque::new(),
            visited_nodes: HashSet::new(),
            dijkstra_state: None,
            dragging_node: None,
            selected_starts: HashSet::new(),
            creation_state: CreationState::None,
            editing_state: EditingState::None,
            next_node_id: 3, // Start ID counter after initial nodes
            show_distances: true,
            step_speed: 0.5,
            last_step_time: 0.0,
        }
    }
}

// --- eframe::App Implementation ---

impl eframe::App for GraphApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint(); // Continuously repaint for animations

        // Handle time-based algorithm stepping
        self.handle_algorithm_step(ctx);

        // Draw the control panel (can influence state for next frame)
        self.draw_control_panel(ctx);

        // Draw the graph in the central panel and handle interactions within it
        egui::CentralPanel::default().show(ctx, |ui| {
            // The interaction area for the graph
            // Allow clicking, dragging, and double-clicking
            let graph_response = ui.interact(
                ui.clip_rect(),
                ui.id().with("graph_interaction"),
                egui::Sense::click_and_drag(), // double_click is checked on response, not sense
            );

            // Handle mouse/keyboard input on the graph area
            self.handle_graph_interaction(ui, &graph_response);

            // Collect shapes to paint - Pass ui here
            let mut shapes = Vec::new();
            self.draw_graph_elements(ui, &mut shapes); // Pass ui

            // Paint all collected shapes
            ui.painter().extend(shapes);

            // Draw temporary elements if in creation/editing mode
            self.draw_temporary_elements(ui);
        });

        // Show modal windows based on current state (e.g., edit edge weight)
        // These should be handled outside the main graph painting area
        self.show_modal_windows(ctx);
    }
}

// --- GraphApp Methods ---

impl GraphApp {
    /// Orchestrates drawing of all graph components.
    /// Accepts `&egui::Ui` to allow text layout.
    fn draw_graph_elements(&self, ui: &egui::Ui, shapes: &mut Vec<Shape>) {
        self.draw_edges(ui, shapes); // Pass ui
        self.draw_nodes(ui, shapes); // Pass ui
        self.draw_distance_labels(ui, shapes); // Pass ui
    }

    /// Draws edges and their weights using direct Shape::Text creation.
    /// Accepts `&egui::Ui` for text layout.
    fn draw_edges(&self, ui: &egui::Ui, shapes: &mut Vec<Shape>) {
        for node in &self.nodes {
            for edge in &node.edges {
                let start = node.position;
                // Find the target node position - handle potential invalid edge.to index defensively
                let end = match self.nodes.get(edge.to) {
                    Some(target_node) => target_node.position,
                    None => {
                        // Should not happen with current node ID management,
                        // but good practice if node deletion were added.
                        // Skip drawing this invalid edge.
                        continue;
                    }
                };

                let color = if edge.visited {
                    COLOR_EDGE_VISITED
                } else {
                    COLOR_EDGE_DEFAULT
                };

                shapes.push(Shape::LineSegment {
                    points: [start, end],
                    stroke: Stroke::new(2.0, color),
                });

                // Draw weight label slightly offset
                let mid = start.lerp(end, 0.5) + edge.direction * 15.0;

                // --- Refactored Text Drawing ---
                let weight_text = format!("{:.1}", edge.weight);
                let font_id = FontId::monospace(12.0);
                let text_color = COLOR_TEXT;

                // Create and layout the LayoutJob to get a Galley
                let layout_job = LayoutJob::simple(weight_text, font_id, text_color, f32::INFINITY);
                // Use ui.fonts to layout the job
                let galley = ui.fonts(|f| f.layout_job(layout_job)); // 直接使用 Arc<Galley>


                // Calculate the top-left position needed to center the galley's bounding box at 'mid'
                let text_size = galley.rect.size();
                let top_left = mid - text_size / 2.0;

                // Push the Shape::Text directly
                shapes.push(Shape::Text(TextShape {
                    underline: Stroke::NONE, // 无下划线
                    override_text_color: None, // 不覆盖文本颜色
                    angle: 0.0, // 文本角度为0(不旋转)
                    pos: top_left,
                    galley, // Galley is Arc<Galley>
                    // color: text_color, // Removed: color is part of LayoutJob or override_text_color
                }));
                // --- End Refactored Text Drawing ---
            }
        }
    }

    /// Draws nodes and their IDs using direct Shape::Text creation.
    /// Accepts `&egui::Ui` for text layout.
    fn draw_nodes(&self, ui: &egui::Ui, shapes: &mut Vec<Shape>) {
        for node in &self.nodes {
            let is_start = self.selected_starts.contains(&node.id);

            let color = if is_start {
                COLOR_NODE_START
            } else {
                match self.selected_algorithm {
                    Algorithm::BFS | Algorithm::DFS => {
                        if self.visited_nodes.contains(&node.id) {
                            COLOR_NODE_VISITED
                        } else {
                            COLOR_NODE_DEFAULT
                        }
                    }
                    Algorithm::Dijkstra => {
                        if let Some(state) = &self.dijkstra_state {
                            // Use visited color for processed nodes in Dijkstra
                            if state.processed_nodes.contains(&node.id) {
                                COLOR_NODE_VISITED
                            } else {
                                COLOR_NODE_DEFAULT
                            }
                        } else {
                            COLOR_NODE_DEFAULT
                        }
                    }
                }
            };

            shapes.push(Shape::Circle(CircleShape {
                center: node.position,
                radius: NODE_RADIUS,
                fill: color,
                stroke: Stroke::new(1.0, Color32::BLACK),
            }));

            // --- Refactored Text Drawing (Node ID) ---
            let text_pos = node.position; // Desired center position for the text
            let node_id_text = node.id.to_string();
            let font_id = FontId::monospace(14.0);
            let text_color = COLOR_TEXT_DARK;

            let layout_job = LayoutJob::simple(node_id_text, font_id, text_color, f32::INFINITY);
            let galley = ui.fonts(|f| f.layout_job(layout_job));

            // Calculate the top-left position needed to center the galley's bounding box at 'text_pos'
            let text_size = galley.rect.size();
            let top_left = text_pos - text_size / 2.0;

            shapes.push(Shape::Text(TextShape {
                pos: top_left,
                galley,
                angle: 0.0, 
                underline: Stroke::NONE, 
                override_text_color: None, // Added missing field
                // color: text_color, // Removed: color is part of LayoutJob or override_text_color
            }));
            // --- End Refactored Text Drawing ---
        }
    }

    /// Draws distance labels for Dijkstra's algorithm using direct Shape::Text creation.
    /// Accepts `&egui::Ui` for text layout.
    fn draw_distance_labels(&self, ui: &egui::Ui, shapes: &mut Vec<Shape>) {
        if self.show_distances {
            if let Some(state) = &self.dijkstra_state {
                // Check bounds defensively
                for id in 0..self.nodes.len().min(state.distances.len()) {
                    let dist = state.distances[id];
                    if dist.is_finite() {
                        // This 'pos' is intended as the LEFT edge of the text block, vertically centered
                        let pos = self.nodes[id].position + Vec2::new(NODE_RADIUS + 5.0, 0.0);

                        // --- Refactored Text Drawing (Distance) ---
                        let dist_text = format!("{:.1}", dist);
                        let font_id = FontId::monospace(12.0);
                        let text_color = COLOR_TEXT_DARK; // 使用深色文本颜色以避免 unwrap 错误

                        let layout_job = LayoutJob::simple(dist_text, font_id, text_color, f32::INFINITY);
                        let galley = ui.fonts(|f| f.layout_job(layout_job));

                        // Calculate the top-left position needed for left-alignment and vertical centering
                        let text_size = galley.rect.size();
                        // The desired position `pos` is the left edge, vertically centered.
                        // The TextShape `pos` is the top-left corner.
                        // So, we need to shift `pos` upwards by half the text height.
                        let top_left = pos - Vec2::new(0.0, text_size.y / 2.0);

                        shapes.push(Shape::Text(TextShape {
                            pos: top_left,
                            galley,
                            override_text_color: Some(text_color),
                            angle: 0.0, // Added missing field
                            underline: Stroke::NONE, // Added missing field
                        }));
                        // --- End Refactored Text Drawing ---
                    }
                }
            }
        }
    }

    /// Draws temporary elements like the line being dragged for edge creation.
    fn draw_temporary_elements(&self, ui: &egui::Ui) {
        if let CreationState::CreatingEdge(from_id) = self.creation_state {
            // Ensure from_id is valid
            if let Some(from_node) = self.nodes.get(from_id) {
                if let Some(pointer_pos) = ui.ctx().pointer_hover_pos() {
                    // Draw a temporary dashed line from the source node to the pointer
                    ui.painter().add(Shape::LineSegment {
                        points: [from_node.position, pointer_pos],
                        stroke: Stroke::new(2.0, Color32::GRAY),
                    });
                }
            } else {
                // Invalid from_id, this state should probably be cleaned up elsewhere,
                // but for drawing, we just skip.
                eprintln!("Warning: CreatingEdge from invalid node ID: {}", from_id);
            }
        }
    }


    /// Handles all mouse and keyboard input events within the graph area.
    fn handle_graph_interaction(&mut self, ui: &egui::Ui, response: &egui::Response) {
        let pointer_pos = response.hover_pos();

        // --- Node Dragging ---
        if response.drag_started() {
            if let Some(pos) = pointer_pos {
                if let Some(node_id) = self.node_at_pos(pos) {
                    self.dragging_node = Some(node_id);
                    // Ensure we are not in creation/editing state if we start dragging
                    self.creation_state = CreationState::None;
                    self.editing_state = EditingState::None;
                }
            }
        }

        if let Some(dragging_node_id) = self.dragging_node {
            if let Some(pos) = pointer_pos {
                // Check bounds defensively
                if let Some(node) = self.nodes.get_mut(dragging_node_id) {
                    node.position = pos;
                    // Recalculate edge directions for connected edges if positions change significantly
                    // For simplicity, we skip this here, but in a real app, you might update edge.direction
                } else {
                    // This node no longer exists, stop dragging
                    self.dragging_node = None;
                }
            }
        }

        if response.drag_released() {
            self.dragging_node = None;
        }

        // --- Node Selection (for start nodes) ---
        // Space + Left Click to toggle start node selection
        if response.clicked() && ui.ctx().input(|i| i.key_pressed(egui::Key::Space)) {
            if let Some(pos) = pointer_pos {
                if let Some(id) = self.node_at_pos(pos) {
                    if !self.selected_starts.insert(id) {
                        self.selected_starts.remove(&id);
                    }
                    // Clear algorithm state when start nodes change
                    self.reset_visits();
                    self.dijkstra_state = None;
                }
            }
        }

        // --- Handle Creation/Editing Initiation and Cancellation ---
        // Don't initiate new actions if currently dragging
        if self.dragging_node.is_none() {

            // Right-click initiates creation (node or edge) or cancels current action
            if response.secondary_clicked() {
                if self.creation_state != CreationState::None || self.editing_state != EditingState::None {
                    // If in creation or editing mode, right-click cancels
                    self.creation_state = CreationState::None;
                    self.editing_state = EditingState::None;
                } else if let Some(pos) = pointer_pos {
                    if let Some(id) = self.node_at_pos(pos) {
                        // Right-clicked on a node -> Start creating an edge from this node
                        self.creation_state = CreationState::CreatingEdge(id);
                    } else {
                        // Right-clicked on empty space -> Start creating a node
                        self.creation_state = CreationState::CreatingNode(pos);
                    }
                }
            }

            // Double-click initiates edge editing (if on an edge)
            if response.double_clicked() {
                if let Some(pos) = pointer_pos {
                    if let Some((from, to)) = self.edge_at_pos(pos, EDGE_INTERACTION_TOLERANCE) {
                        // If in edge creation mode, double click on the *same* edge might be ambiguous.
                        // Assume editing takes priority over creation for double click.
                        self.creation_state = CreationState::None; // Cancel creation if active
                        self.editing_state = EditingState::EditingEdgeWeight(from, to); // Initiate editing
                    }
                }
            }

            // Left-click finalizes edge creation or cancels creation/editing if clicked elsewhere
            if response.clicked() && !ui.ctx().input(|i| i.key_pressed(egui::Key::Space)) { // Exclude Space+Click (handled above)
                match self.creation_state {
                    CreationState::CreatingEdge(from_id) => {
                        if let Some(pos) = pointer_pos {
                            if let Some(to_id) = self.node_at_pos(pos) {
                                // Clicked on a node -> Finalize edge creation
                                if from_id != to_id {
                                    // Check if edge already exists and update it, or add new
                                    let edge_exists = self.nodes.get(from_id) // Defensive get
                                        .and_then(|node| node.edges.iter().find(|e| e.to == to_id))
                                        .is_some();

                                    if !edge_exists {
                                        // Ensure both nodes are still valid
                                        if from_id < self.nodes.len() && to_id < self.nodes.len() {
                                            let from_pos = self.nodes[from_id].position;
                                            let to_pos = self.nodes[to_id].position;
                                            let direction = (to_pos - from_pos).normalized().rot90();
                                            // Add edge to the 'from' node's edge list
                                            // This requires mutable borrow of the 'from' node
                                            if let Some(from_node_mut) = self.nodes.get_mut(from_id) {
                                                from_node_mut.edges.push(Edge {
                                                    to: to_id,
                                                    weight: 1.0, // Default weight
                                                    visited: false,
                                                    direction,
                                                });
                                            }
                                        } else {
                                            eprintln!("Warning: Attempted to create edge with invalid node ID(s): {} -> {}", from_id, to_id);
                                        }
                                    }
                                }
                                self.creation_state = CreationState::None; // End creation state
                            } else {
                                // Clicked on empty space -> Cancel edge creation
                                self.creation_state = CreationState::None; // End creation state
                            }
                        } else {
                            // No pointer pos (shouldn't happen with response.clicked but defensive)
                            self.creation_state = CreationState::None;
                        }
                    }
                    CreationState::CreatingNode(_) => {
                        // Node creation window is modal, handled by show_modal_windows.
                        // Clicking on the graph while this window is open *doesn't* cancel it automatically.
                        // Cancellation happens via the window buttons or closing the window.
                    }
                    CreationState::None => {
                        // No creation action in progress, check if we need to cancel editing by clicking elsewhere
                        if self.editing_state != EditingState::None {
                            // Check if the click was NOT on an edge
                            if let Some(pos) = pointer_pos {
                                if self.edge_at_pos(pos, EDGE_INTERACTION_TOLERANCE).is_none() {
                                    // Clicked somewhere NOT on an edge
                                    self.editing_state = EditingState::None; // Cancel editing
                                }
                            } else {
                                // No pointer pos, cancel editing
                                self.editing_state = EditingState::None;
                            }
                        }
                    }
                }
            }
        } // End if not dragging
    }

    /// Shows modal windows for creation or editing based on state.
    // 显示用于创建或编辑的模态窗口（基于状态）
    fn show_modal_windows(&mut self, ctx: &egui::Context) {
        // Node Creation Window
        // 节点创建窗口
        if let CreationState::CreatingNode(pos) = self.creation_state {
            let mut open = true; // Window is open by default when state is active // 状态激活时窗口默认打开
            egui::Window::new("Create New Node") // 创建新节点
                .collapsible(false)
                .resizable(false)
                .open(&mut open) // Allow closing with 'X' button // 允许使用“X”按钮关闭
                .show(ctx, |ui| {
                    ui.label(format!("Create node {} at position ({:.1}, {:.1})", self.next_node_id, pos.x, pos.y)); // 将在位置 ... 创建节点 ...
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Confirm").clicked() { // 确认
                            self.nodes.push(Node {
                                id: self.next_node_id,
                                position: pos,
                                edges: Vec::new(),
                            });
                            self.next_node_id += 1;
                            self.creation_state = CreationState::None; // End state on confirm // 确认后结束状态
                        }
                        if ui.button("Cancel").clicked() { // 取消
                            self.creation_state = CreationState::None; // End state on cancel // 取消后结束状态
                        }
                    });
                });
            // If the user closed the window using the 'X' button
            // 如果用户使用“X”按钮关闭了窗口
            if !open {
                self.creation_state = CreationState::None;
            }
        }

        // Edge Editing Window
        // 边编辑窗口
        if let EditingState::EditingEdgeWeight(from_id, to_id) = self.editing_state {
            // Find the edge we are editing. Use indices defensively.
            // 查找我们正在编辑的边。谨慎使用索引。
            let edge_option = self.nodes.get_mut(from_id).and_then(|node|
                node.edges.iter_mut().find(|e| e.to == to_id)
            );

            if let Some(edge) = edge_option {
                let mut open = true; // Window is open by default when state is active // 状态激活时窗口默认打开
                let mut current_weight = edge.weight; // Bind weight to a local mutable variable for the slider // 将权重绑定到局部可变变量以用于滑块

                egui::Window::new("Edit Edge Weight") // 编辑边权重
                    .collapsible(false)
                    .resizable(false)
                    .open(&mut open) // Allow closing with 'X' button // 允许使用“X”按钮关闭
                    .show(ctx, |ui| {
                        ui.label(format!("Edit weight from Node {} to Node {}", from_id, to_id)); // 编辑节点 ... 到节点 ... 的权重
                        ui.add(egui::Slider::new(&mut current_weight, 0.1..=10.0).text("Weight")); // 权重 // Slider updates current_weight // 滑块更新 current_weight
                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button("Save").clicked() { // 保存
                                // Find the edge again to update its actual weight from `current_weight`
                                // 再次查找边以从 `current_weight` 更新其实际权重
                                if let Some(edge_to_update) = self.nodes.get_mut(from_id).and_then(|node|
                                    node.edges.iter_mut().find(|e| e.to == to_id)
                                ) {
                                    edge_to_update.weight = current_weight;
                                }
                                self.editing_state = EditingState::None; // End state on save // 保存后结束状态
                            }
                            if ui.button("Cancel").clicked() { // 取消
                                // current_weight changes are discarded by not copying back to edge
                                // current_weight 的更改不会复制回边，因此被丢弃
                                self.editing_state = EditingState::None; // End state on cancel // 取消后结束状态
                            }
                        });
                    });
                // If the user closed the window using the 'X' button
                // 如果用户使用“X”按钮关闭了窗口
                if !open {
                    self.editing_state = EditingState::None;
                }
            } else {
                // Edge no longer exists (e.g., node deleted)? Cancel editing state.
                // 边不再存在（例如，节点已删除）？取消编辑状态。
                self.editing_state = EditingState::None;
            }
        }
    }


    /// Handles the time-based stepping of the selected algorithm.
    // 处理所选算法的基于时间的步进。
    fn handle_algorithm_step(&mut self, ctx: &egui::Context) {
        let current_time = ctx.input(|i| i.time); // Use input().time for consistent delta time
        let time_since_last_step = current_time - self.last_step_time;

        // Only step if enough time has passed and there is work to do
        if time_since_last_step >= self.step_speed as f64 {
            let stepped = match self.selected_algorithm {
                Algorithm::BFS => self.step_bfs(),
                Algorithm::DFS => self.step_dfs(),
                Algorithm::Dijkstra => self.step_dijkstra(),
            };

            if stepped {
                // Update last step time only if a step actually occurred
                self.last_step_time = current_time;
            }
        }
    }

    /// Performs one step of the Breadth-First Search algorithm.
    /// Returns true if a step was taken, false if the queue was empty.
    // 执行广度优先搜索算法的一步。
    // 如果执行了一步则返回 true，如果队列为空则返回 false。
    fn step_bfs(&mut self) -> bool {
        // Get a snapshot of node count before borrowing mutably
        let num_nodes = self.nodes.len();
        while let Some(current_id) = self.work_queue_stack.pop_front() {
            // Check bounds defensively before borrowing
            if current_id < num_nodes {
                if let Some(current_node) = self.nodes.get_mut(current_id) {
                    for edge in &mut current_node.edges {
                        // Check bounds for target node before using it
                        if edge.to < num_nodes && !self.visited_nodes.contains(&edge.to) {
                            edge.visited = true;
                            self.visited_nodes.insert(edge.to);
                            self.work_queue_stack.push_back(edge.to);
                        }
                    }
                    return true; // Step was taken (a valid node was processed)
                }
                // If get_mut failed (shouldn't happen with valid ID and num_nodes check),
                // it's an invalid ID in the queue. The loop continues to pop the next item.
            }
            // If the ID was out of bounds initially, the loop continues to pop the next item.
        }
        false // Queue was empty or only contained invalid IDs
    }

    /// Performs one step of the Depth-First Search algorithm.
    /// Returns true if a step was taken, false if the stack was empty.
    fn step_dfs(&mut self) -> bool {
        // Get a snapshot of node count before borrowing mutably
        let num_nodes = self.nodes.len();
        while let Some(current_id) = self.work_queue_stack.pop_back() {
            // Check bounds defensively before borrowing
            if current_id < num_nodes {
                if let Some(current_node) = self.nodes.get_mut(current_id) {
                    // Process neighbors in reverse order to match typical recursive DFS behavior
                    // (or just consistent order for visualization)
                    for edge in current_node.edges.iter_mut().rev() {
                        // Check bounds for target node before using it
                        if edge.to < num_nodes && !self.visited_nodes.contains(&edge.to) {
                            edge.visited = true;
                            self.visited_nodes.insert(edge.to);
                            self.work_queue_stack.push_back(edge.to);
                        }
                    }
                    return true; // Step was taken (a valid node was processed)
                }
                // If get_mut failed, the loop continues.
            }
            // Invalid ID in stack, the loop continues.
        }
        false // Stack was empty or only contained invalid IDs
    }

    /// Performs one step of Dijkstra's algorithm.
    /// Returns true if a step was taken (a node was processed), false if the heap is empty.
    fn step_dijkstra(&mut self) -> bool {
        if let Some(state) = &mut self.dijkstra_state {
            // Get a snapshot of node count before borrowing mutably or immutably inside the loop
            let num_nodes = self.nodes.len();

            while let Some(current) = state.heap.pop() {
                // If we found a shorter path already, ignore this outdated entry
                // Also check if the node ID is valid before accessing state.distances
                if current.id >= num_nodes || current.distance > state.distances.get(current.id).copied().unwrap_or(INFINITY) {
                    continue;
                }

                // Node is now processed (final shortest distance found)
                state.processed_nodes.insert(current.id);

                // Relax edges from the processed node
                // Use get() for immutable borrow inside the loop
                if let Some(current_node) = self.nodes.get(current.id) {
                    for edge in &current_node.edges {
                        // Check bounds defensively for target node
                        if edge.to < num_nodes {
                            let new_dist = current.distance + edge.weight;
                            if new_dist < state.distances[edge.to] {
                                state.distances[edge.to] = new_dist;
                                state.predecessors.insert(edge.to, current.id); // Store predecessor
                                state.heap.push(DijkstraNode {
                                    id: edge.to,
                                    distance: new_dist,
                                });
                            }
                        }
                    }
                } // else: invalid current_id (already handled by the continue check above)

                return true; // A valid node was processed
            }
        }
        false // Heap was empty or only contained outdated/invalid entries
    }

    /// Draws the control panel with algorithm selection, start nodes, etc.
    fn draw_control_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("controls").show(ctx, |ui| {
            ui.heading("Controls"); // Control Panel
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Algorithm:"); // Algorithm:
                egui::ComboBox::from_id_source("algo_select")
                    .selected_text(format!("{:?}", self.selected_algorithm))
                    .show_ui(ui, |ui| {
                        // Use temporary variable to detect changes
                        let mut selected = self.selected_algorithm;
                        if ui.selectable_value(
                            &mut selected,
                            Algorithm::BFS,
                            "Breadth-First Search (BFS)", // Breadth-First Search (BFS)
                        ).clicked() {
                            self.selected_algorithm = selected;
                            self.reset_visits(); // Reset state when algorithm changes
                        }
                        if ui.selectable_value(
                            &mut selected,
                            Algorithm::DFS,
                            "Depth-First Search (DFS)", // Depth-First Search (DFS)
                        ).clicked() {
                            self.selected_algorithm = selected;
                            self.reset_visits(); // Reset state when algorithm changes
                        }
                        if ui.selectable_value(
                            &mut selected,
                            Algorithm::Dijkstra,
                            "Dijkstra's Algorithm", // Dijkstra's Algorithm
                        ).clicked() {
                            self.selected_algorithm = selected;
                            self.reset_visits(); // Reset state when algorithm changes
                        }
                    });
            });

            ui.separator();
            ui.checkbox(&mut self.show_distances, "Show Distances (Dijkstra)"); // Show Distances (Dijkstra)
            ui.add(egui::Slider::new(&mut self.step_speed, 0.05..=2.0).text("Step Speed (s/step)")); // Step Speed (s/step) // Allow faster steps

            ui.separator();
            ui.label("Selected Start Nodes (Space + Left Click Node):"); // Selected Start Nodes (Space + Left Click Node):
            if self.selected_starts.is_empty() {
                ui.label("None"); // None
            } else {
                let mut starts: Vec<_> = self.selected_starts.iter().copied().collect();
                starts.sort(); // Sort for consistent display
                for (_i, start) in starts.into_iter().enumerate() {
                    ui.label(format!("Node {}", start)); // Node {}
                }
            }

            ui.separator();
            // Disable start button if no start nodes are selected or algorithm is already running
            let algorithm_running = !self.work_queue_stack.is_empty() || self.dijkstra_state.as_ref().map_or(false, |state| !state.heap.is_empty());
            if ui.add_enabled(!self.selected_starts.is_empty() && !algorithm_running, egui::Button::new("Start Algorithm")).clicked() { // Start Algorithm
                // Reset current run state, but keep selected starts
                self.reset_visits();
                match self.selected_algorithm {
                    Algorithm::BFS | Algorithm::DFS => {
                        // Ensure selected starts are valid node IDs
                        for &start in &self.selected_starts {
                            if start < self.nodes.len() {
                                self.work_queue_stack.push_back(start); // BFS/DFS start node is added once
                                self.visited_nodes.insert(start);
                            } else {
                                eprintln!("Warning: Selected start node {} is out of bounds. Ignoring.", start);
                            }
                        }
                    }
                    Algorithm::Dijkstra => self.init_dijkstra(), // init_dijkstra handles start node validity
                }
            }

            if ui.button("Reset").clicked() { // Reset
                self.reset_visits();
                self.selected_starts.clear(); // Also clear selected starts on full reset
            }

            ui.separator();
            ui.label("Interaction Guide:"); // Interaction Guide:
            ui.label(" - Left-click and drag node to move"); // - Left-click and drag node to move
            ui.label(" - Space + Left-click node: Toggle start node"); // - Space + Left-click node: Toggle start node
            ui.label(" - Right-click empty space: Create new node"); // - Right-click empty space: Create new node
            ui.label(" - Right-click node: Start creating edge"); // - Right-click node: Start creating edge
            ui.label("   (Drag to target node and left-click to finish)"); //   (Drag to target node and left-click to finish)
            ui.label(" - Double-click edge: Edit weight"); // - Double-click edge: Edit weight
            ui.label(" - Right-click or click empty space (not on node/edge): Cancel creation/editing"); // - Right-click or click empty space (not on node/edge): Cancel creation/editing
        });
    }

    /// Resets the algorithm state (visited nodes, queue/stack, Dijkstra state).
    /// Does NOT clear selected start nodes.
    // Resets the algorithm state (visited nodes, queue/stack, Dijkstra state).
    // Does NOT clear selected start nodes.
    fn reset_visits(&mut self) {
        self.visited_nodes.clear();
        self.work_queue_stack.clear();
        // Reset edge visited flags
        for node in &mut self.nodes {
            for edge in &mut node.edges {
                edge.visited = false;
            }
        }
        // Dijkstra specific reset
        self.dijkstra_state = None;
    }

    /// Initializes the state for Dijkstra's algorithm based on selected start nodes.
    fn init_dijkstra(&mut self) {
        if self.selected_starts.is_empty() {
            return; // Cannot start Dijkstra without start nodes
        }

        let num_nodes = self.nodes.len();
        let mut distances = vec![INFINITY; num_nodes];
        let mut heap = BinaryHeap::new();
        let predecessors = HashMap::new();
        let processed_nodes = HashSet::new();

        for &start in &self.selected_starts {
            // Ensure start node ID is valid before adding to distances/heap
            if start < num_nodes {
                distances[start] = 0.0;
                heap.push(DijkstraNode {
                    id: start,
                    distance: 0.0,
                });
                // Note: Dijkstra doesn't mark start nodes as 'visited'/'processed' initially,
                // they get processed when popped from the heap.
            } else {
                eprintln!("Warning: Selected start node {} is out of bounds. Ignoring.", start);
            }
        }

        self.dijkstra_state = Some(DijkstraState {
            distances,
            heap,
            predecessors,
            processed_nodes,
        });
    }

    /// Finds the ID of a node at a given position, if any.
    fn node_at_pos(&self, pos: Pos2) -> Option<usize> {
        self.nodes
            .iter()
            .find(|n| (n.position - pos).length() < INTERACTION_RADIUS)
            .map(|n| n.id)
    }

    /// Finds the edge between two nodes and returns a mutable reference.
    /// Assumes node IDs are valid indices.
    fn get_edge_mut(&mut self, from: usize, to: usize) -> Option<&mut Edge> {
        // Check bounds defensively
        self.nodes.get_mut(from).and_then(|node|
            node.edges.iter_mut().find(|e| e.to == to)
        )
    }


    /// Finds if the given position is near any edge, and returns the (from_id, to_id) pair.
    fn edge_at_pos(&self, pos: Pos2, tolerance: f32) -> Option<(usize, usize)> {
        for node1 in &self.nodes {
            for edge in &node1.edges {
                // Check bounds for target node defensively
                if let Some(node2) = self.nodes.get(edge.to) {
                    let p1 = node1.position;
                    let p2 = node2.position;

                    // Calculate distance from 'pos' to the line segment [p1, p2]
                    let line_vec = p2 - p1;
                    let point_vec = pos - p1;
                    let line_len_sq = line_vec.length_sq();

                    let t = if line_len_sq == 0.0 { // Handle zero-length edges (shouldn't happen with unique nodes but defensuve)
                        0.0
                    } else {
                        point_vec.dot(line_vec) / line_len_sq
                    };

                    // Clamp t to the [0, 1] range to check distance to the segment, not the infinite line
                    let t_clamped = t.max(0.0).min(1.0);

                    let projection = p1 + line_vec * t_clamped;
                    let distance = (pos - projection).length();

                    if distance < tolerance {
                        // Return the actual from_id and to_id
                        return Some((node1.id, edge.to));
                    }
                }
            }
        }
        None // No edge found near the position
    }
}

// --- Main Function ---

fn main() {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Graph Algorithm Visualization", // Graph Algorithm Visualization
        options,
        Box::new(|_| Box::new(GraphApp::default())),
    )
        .unwrap();
}
