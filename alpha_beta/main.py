import streamlit as st
import time
from typing import List, Tuple, Optional

class BrickGame:
    """
    Zero-sum pile-splitting game with 7 bricks initially.
    Players alternate turns splitting one pile into two unequal piles.
    A player wins when it's their turn and all piles have 1 or 2 bricks.
    """
    
    def __init__(self, initial_piles=None):
        if initial_piles is None:
            self.initial_piles = [7]  # Start with one pile of 7 bricks
        else:
            self.initial_piles = initial_piles
    
    def is_terminal(self, piles):
        """Check if the game is over (all piles have 1 or 2 bricks)"""
        return all(pile <= 2 for pile in piles)
    
    def get_possible_moves(self, piles):
        """
        Get all possible moves: split any pile > 2 into two unequal piles
        Returns list of tuples: (pile_index, split1, split2)
        """
        moves = []
        for i, pile in enumerate(piles):
            if pile > 2:
                # Generate all possible unequal splits
                for split1 in range(1, pile // 2 + 1):
                    split2 = pile - split1
                    if split1 != split2:  # Must be unequal
                        moves.append((i, split1, split2))
        return moves
    
    def apply_move(self, piles, move):
        """
        Apply a move to the current game state
        Returns new piles list after applying the move
        """
        pile_index, split1, split2 = move
        new_piles = piles.copy()
        new_piles[pile_index:pile_index+1] = [split1, split2]  # Replace pile with two splits
        return sorted(new_piles, reverse=True)  # Sort for consistency
    
    def evaluate(self, piles, is_maximizing_player):
        """
        Evaluate the game state:
        - If terminal state and it's maximizing player's turn, they lose (-1)
        - If terminal state and it's minimizing player's turn, maximizing player wins (+1)
        - Non-terminal states return 0
        """
        if self.is_terminal(piles):
            return -1 if is_maximizing_player else 1
        return 0
    
    def minimax_alpha_beta(self, piles, depth, alpha, beta, is_maximizing_player, max_depth=10):
        """
        Minimax algorithm with alpha-beta pruning
        
        Args:
            piles: Current game state (list of pile sizes)
            depth: Current depth in the search tree
            alpha: Best value for maximizing player
            beta: Best value for minimizing player
            is_maximizing_player: True if current player is maximizing
            max_depth: Maximum search depth
        
        Returns:
            Best evaluation score for current position
        """
        
        # Terminal condition: game over or max depth reached
        if self.is_terminal(piles) or depth >= max_depth:
            return self.evaluate(piles, is_maximizing_player)
        
        if is_maximizing_player:
            max_eval = float('-inf')
            
            for move in self.get_possible_moves(piles):
                new_piles = self.apply_move(piles, move)
                eval_score = self.minimax_alpha_beta(
                    new_piles, depth + 1, alpha, beta, False, max_depth
                )
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
                    
            return max_eval
        
        else:  # Minimizing player
            min_eval = float('inf')
            
            for move in self.get_possible_moves(piles):
                new_piles = self.apply_move(piles, move)
                eval_score = self.minimax_alpha_beta(
                    new_piles, depth + 1, alpha, beta, True, max_depth
                )
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
                    
            return min_eval
    
    def get_best_move(self, piles, is_maximizing_player=True):
        """
        Find the best move for the current player
        
        Returns:
            Tuple of (best_move, best_score)
        """
        best_move = None
        best_score = float('-inf') if is_maximizing_player else float('inf')
        
        for move in self.get_possible_moves(piles):
            new_piles = self.apply_move(piles, move)
            score = self.minimax_alpha_beta(
                new_piles, 0, float('-inf'), float('inf'), not is_maximizing_player
            )
            
            if is_maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        
        return best_move, best_score


def display_piles(piles: List[int], title: str = "Current Game State"):
    """Display the piles visually using Streamlit"""
    st.subheader(title)
    
    # Create columns for each pile
    cols = st.columns(len(piles) + 1)
    
    for i, pile_size in enumerate(piles):
        with cols[i]:
            st.write(f"**Pile {i+1}**")
            # Create visual representation of bricks
            brick_display = "🧱 " * pile_size
            st.write(brick_display)
            st.write(f"Size: {pile_size}")


def display_moves(game: BrickGame, piles: List[int]):
    """Display available moves as clickable buttons"""
    moves = game.get_possible_moves(piles)
    
    if not moves:
        st.error("No valid moves available!")
        return None
    
    st.subheader("Your Move")
    st.write("Click on a pile to see splitting options:")
    
    # Group moves by pile
    moves_by_pile = {}
    for move in moves:
        pile_idx, split1, split2 = move
        if pile_idx not in moves_by_pile:
            moves_by_pile[pile_idx] = []
        moves_by_pile[pile_idx].append((split1, split2))
    
    # Create buttons for each pile that can be split
    for pile_idx in moves_by_pile:
        with st.container():
            st.write(f"**Pile {pile_idx+1} ({piles[pile_idx]} bricks)** - Split into:")
            
            # Create columns for split options
            cols = st.columns(len(moves_by_pile[pile_idx]))
            
            for i, (split1, split2) in enumerate(moves_by_pile[pile_idx]):
                with cols[i]:
                    button_text = f"{split1} + {split2}"
                    if st.button(button_text, key=f"move_{pile_idx}_{split1}_{split2}"):
                        return (pile_idx, split1, split2)
    
    return None


def display_ai_analysis(game: BrickGame, piles: List[int]):
    """Display AI analysis of the current position"""
    st.subheader("🤖 AI Analysis")
    
    with st.expander("View AI Analysis", expanded=False):
        moves = game.get_possible_moves(piles)
        
        if moves:
            st.write("**Move Evaluation:**")
            
            # Analyze each possible move
            for move in moves:
                pile_idx, split1, split2 = move
                new_piles = game.apply_move(piles, move)
                score = game.minimax_alpha_beta(
                    new_piles, 0, float('-inf'), float('inf'), False
                )
                
                st.write(f"Split Pile {pile_idx+1} ({piles[pile_idx]}) → {split1} + {split2}: Score = {score}")
            
            # Show best move
            best_move, best_score = game.get_best_move(piles, is_maximizing_player=True)
            if best_move:
                pile_idx, split1, split2 = best_move
                st.success(f"**AI's Best Move:** Split Pile {pile_idx+1} ({piles[pile_idx]}) → {split1} + {split2} (Score: {best_score})")


def play_game_tab():
    """Main game playing interface"""
    st.header("🎮 Play Against AI")
    
    # Initialize game state
    if 'game_piles' not in st.session_state:
        st.session_state.game_piles = [7]
        st.session_state.current_player = 'human'  # 'human' or 'ai'
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.move_history = []
    
    game = BrickGame()
    
    # Simple game controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔄 New Game", type="primary"):
            st.session_state.game_piles = [7]
            st.session_state.current_player = 'human'
            st.session_state.game_over = False
            st.session_state.winner = None
            st.session_state.move_history = []
            st.rerun()
    
    with col2:
        show_ai_help = st.checkbox("💡 Show AI hints")
    
    # Display current game state
    display_piles(st.session_state.game_piles)
    
    # Check for game over condition (when all piles have 1 or 2 bricks)
    if game.is_terminal(st.session_state.game_piles):
        if not st.session_state.game_over:
            st.session_state.game_over = True
            # The player who CANNOT move loses (opposite of original)
            st.session_state.winner = 'ai' if st.session_state.current_player == 'human' else 'human'
    
    if st.session_state.game_over:
        if st.session_state.winner == 'human':
            st.success("🎉 You won! The AI had no moves left!")
        else:
            st.error("😔 You lost! You had no moves left!")
        return
    
    # Current player indicator
    if st.session_state.current_player == 'human':
        st.info("🙋 Your turn!")
        
        # Show AI hints if enabled
        if show_ai_help:
            with st.expander("💡 AI Hint", expanded=True):
                best_move, best_score = game.get_best_move(st.session_state.game_piles, is_maximizing_player=True)
                if best_move:
                    pile_idx, split1, split2 = best_move
                    st.write(f"🤖 AI suggests: Split Pile {pile_idx+1} into {split1} + {split2}")
        
        # Handle human move
        selected_move = display_moves(game, st.session_state.game_piles)
        
        if selected_move:
            pile_idx, split1, split2 = selected_move
            move_desc = f"Split Pile {pile_idx+1} ({st.session_state.game_piles[pile_idx]}) → {split1} + {split2}"
            
            st.session_state.game_piles = game.apply_move(st.session_state.game_piles, selected_move)
            st.session_state.move_history.append(("You", move_desc))
            st.session_state.current_player = 'ai'
            st.rerun()
    
    else:  # AI turn
        st.info("🤖 AI is thinking...")
        
        # AI makes move
        best_move, score = game.get_best_move(st.session_state.game_piles, is_maximizing_player=True)
        
        if best_move:
            pile_idx, split1, split2 = best_move
            move_desc = f"Split Pile {pile_idx+1} ({st.session_state.game_piles[pile_idx]}) → {split1} + {split2}"
            
            # Add some suspense
            time.sleep(1)
            
            st.session_state.game_piles = game.apply_move(st.session_state.game_piles, best_move)
            st.session_state.move_history.append(("AI", move_desc))
            st.session_state.current_player = 'human'
            
            st.success(f"AI: {move_desc}")
            time.sleep(1)
            st.rerun()


def analysis_tab():
    """Analysis and demonstration interface"""
    st.header("🧠 Algorithm Analysis")
    
    st.write("""
    This tab demonstrates the minimax algorithm with alpha-beta pruning used by the AI.
    You can analyze different game positions and see how the AI evaluates them.
    """)
    
    # Custom position setup
    st.subheader("🔧 Setup Custom Position")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Preset Positions:**")
        preset_positions = {
            "Starting position": [7],
            "Early game": [5, 2],
            "Mid game": [4, 3],
            "Complex position": [3, 2, 2],
            "Near endgame": [3, 2, 1],
            "Late game": [2, 2, 2, 1],
            "Almost terminal": [2, 1, 1, 1]
        }
        
        selected_preset = st.selectbox("Choose a preset:", list(preset_positions.keys()))
        analysis_piles = preset_positions[selected_preset]
    
    with col2:
        st.write("**Custom Position:**")
        custom_input = st.text_input("Enter pile sizes (comma-separated):", "7")
        
        if st.button("Use Custom Position"):
            try:
                analysis_piles = [int(x.strip()) for x in custom_input.split(',') if x.strip()]
                if not analysis_piles:
                    st.error("Please enter valid pile sizes")
                else:
                    st.success(f"Using custom position: {analysis_piles}")
            except ValueError:
                st.error("Please enter valid numbers separated by commas")
    
    # Display analysis
    st.subheader("📊 Position Analysis")
    display_piles(analysis_piles, "Analyzing Position")
    
    game = BrickGame()
    
    if game.is_terminal(analysis_piles):
        st.success("🏁 This is a terminal position - game over!")
    else:
        # Analyze all possible moves
        moves = game.get_possible_moves(analysis_piles)
        
        st.write("**Move Evaluation:**")
        
        move_analysis = []
        for move in moves:
            pile_idx, split1, split2 = move
            new_piles = game.apply_move(analysis_piles, move)
            
            # Evaluate for both players
            score_max = game.minimax_alpha_beta(new_piles, 0, float('-inf'), float('inf'), False)
            score_min = game.minimax_alpha_beta(new_piles, 0, float('-inf'), float('inf'), True)
            
            move_analysis.append({
                'move': f"Split Pile {pile_idx+1} ({analysis_piles[pile_idx]}) → {split1} + {split2}",
                'resulting_piles': new_piles,
                'score_for_maximizer': score_max,
                'score_for_minimizer': score_min
            })
        
        # Display move analysis
        for i, analysis in enumerate(move_analysis):
            with st.expander(f"Move {i+1}: {analysis['move']}", expanded=i == 0):
                st.write(f"**Resulting position:** {analysis['resulting_piles']}")
                st.write(f"**Score for Maximizing Player:** {analysis['score_for_maximizer']}")
                st.write(f"**Score for Minimizing Player:** {analysis['score_for_minimizer']}")
                
                # Show best moves
                if analysis['score_for_maximizer'] == max(a['score_for_maximizer'] for a in move_analysis):
                    st.success("✅ Best move for Maximizing Player")
                if analysis['score_for_minimizer'] == min(a['score_for_minimizer'] for a in move_analysis):
                    st.success("✅ Best move for Minimizing Player")


def main():
    st.set_page_config(
        page_title="Brick Game AI",
        page_icon="🧱",
        layout="wide"
    )
    
    st.title("🧱 Pile-Splitting Brick Game")
    
    # Simple game rules at the top
    st.info("**Goal:** Force your opponent into a position where all piles have 1-2 bricks (they lose!)")
    st.write("**Rules:** Split any pile of 3+ bricks into two *unequal* piles. Avoid being the last to move!")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎮 Play Game", "🧠 Algorithm Analysis"])
    
    with tab1:
        play_game_tab()
    
    with tab2:
        analysis_tab()
    
    # Simplified sidebar
    with st.sidebar:
        st.header("🎯 Quick Tips")
        st.write("""
        - Click buttons to split piles
        - Try to leave your opponent with no good moves
        - Use AI hints to learn strategy
        """)
        
        if st.session_state.get('move_history'):
            st.header("📝 Recent Moves")
            for i, (player, move_desc) in enumerate(st.session_state.move_history[-3:]):
                st.write(f"**{player}:** {move_desc.split(' → ')[1] if ' → ' in move_desc else move_desc}")


if __name__ == "__main__":
    main()