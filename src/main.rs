use std::fs;
use std::process;

use clap::{Parser, Subcommand};

use seedac::codec;

#[derive(Parser)]
#[command(name = "seedac", about = "Seeded Arithmetic Coder — PPM-based compression")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a file
    #[command(name = "c")]
    Compress {
        /// Input file path
        input: String,
        /// Output file path (default: input.seed)
        #[arg(short, long)]
        output: Option<String>,
        /// Max PPM order (default: 4)
        #[arg(long)]
        order: Option<u8>,
        /// Seed: 'auto' (default), a name (english, k8s), or numeric ID
        #[arg(long, default_value = "auto")]
        seed: String,
        /// Path to a .seedmodel recipe file
        #[arg(long)]
        recipe: Option<String>,
    },
    /// Decompress a .seed file
    #[command(name = "d")]
    Decompress {
        /// Input .seed file path
        input: String,
        /// Output file path (default: strip .seed extension)
        #[arg(short, long)]
        output: Option<String>,
        /// Path to a .seedmodel recipe file
        #[arg(long)]
        recipe: Option<String>,
    },
    /// List available seed models
    Seeds,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress {
            input,
            output,
            order,
            seed,
            recipe,
        } => {
            if recipe.is_some() && seed != "auto" {
                eprintln!("Error: cannot use --seed and --recipe together");
                process::exit(1);
            }

            let data = fs::read(&input).unwrap_or_else(|e| {
                eprintln!("Error reading {}: {}", input, e);
                process::exit(1);
            });

            let max_order = order.unwrap_or(4);

            let compressed = if let Some(recipe_path) = recipe {
                let (recipe_order, recipe_counts) =
                    codec::read_recipe(std::path::Path::new(&recipe_path)).unwrap_or_else(|e| {
                        eprintln!("Error reading recipe: {}", e);
                        process::exit(1);
                    });
                let actual_order = order.unwrap_or(recipe_order);
                codec::encode(&data, 0, actual_order, Some(&recipe_counts), false)
            } else {
                match codec::resolve_seed(&seed) {
                    Ok((seed_id, seed_counts, is_auto)) => {
                        if is_auto {
                            codec::encode(&data, 0, max_order, None, true)
                        } else {
                            codec::encode(&data, seed_id, max_order, seed_counts.as_ref(), false)
                        }
                    }
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        process::exit(1);
                    }
                }
            };

            let out_path = output.unwrap_or_else(|| format!("{}.seed", input));
            fs::write(&out_path, &compressed).unwrap_or_else(|e| {
                eprintln!("Error writing {}: {}", out_path, e);
                process::exit(1);
            });

            let ratio = if !data.is_empty() {
                compressed.len() as f64 / data.len() as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "{} -> {} bytes ({:.1}%)",
                data.len(),
                compressed.len(),
                ratio
            );
        }
        Commands::Decompress {
            input,
            output,
            recipe,
        } => {
            let compressed = fs::read(&input).unwrap_or_else(|e| {
                eprintln!("Error reading {}: {}", input, e);
                process::exit(1);
            });

            let recipe_counts;
            let seed_counts_ref = if let Some(recipe_path) = recipe {
                let (_order, counts) =
                    codec::read_recipe(std::path::Path::new(&recipe_path)).unwrap_or_else(|e| {
                        eprintln!("Error reading recipe: {}", e);
                        process::exit(1);
                    });
                recipe_counts = counts;
                Some(&recipe_counts)
            } else {
                None
            };

            let data = codec::decode(&compressed, seed_counts_ref).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                process::exit(1);
            });

            let out_path = output.unwrap_or_else(|| {
                if input.ends_with(".seed") {
                    input[..input.len() - 5].to_string()
                } else {
                    format!("{}.out", input)
                }
            });

            fs::write(&out_path, &data).unwrap_or_else(|e| {
                eprintln!("Error writing {}: {}", out_path, e);
                process::exit(1);
            });

            println!("{} -> {} bytes", compressed.len(), data.len());
        }
        Commands::Seeds => {
            let seeds = codec::list_seeds();
            if seeds.is_empty() {
                println!("No seed models found in seeds/");
                return;
            }
            println!("{:<20} {:>4} {:>6}", "name", "id", "order");
            println!("{}", "-".repeat(40));
            for seed in seeds {
                let total_contexts: usize = seed
                    .counts
                    .values()
                    .map(|oc| oc.len())
                    .sum();
                println!(
                    "{:<20} {:>4} {:>6} {:>6} contexts",
                    seed.name, seed.seed_id, seed.max_order, total_contexts
                );
            }
        }
    }
}
