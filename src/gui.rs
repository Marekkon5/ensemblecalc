use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use std::collections::HashMap;
use anyhow::Error;
use argmin::core::{State, Solver, Problem};
use argmin::solver::neldermead::NelderMead;
use crossbeam_channel::Receiver;
use eframe::{App, NativeOptions};
use egui::{Context, SidePanel, Slider, Ui, ProgressBar, Window};
use egui::plot::{Plot, Line, PlotPoints, Legend};
use humansize::DECIMAL;
use rand::Rng;
use rfd::FileDialog;

use crate::TrackSetSet;

pub fn start_gui() {
    let mut native_options = NativeOptions::default();
    native_options.transparent = true;
    eframe::run_native(
        "Ensemble Calc", 
        native_options, 
        Box::new(|_cc| Box::new(EnsembleCalcGUI::default()))
    ).expect("GUI Failed");
}

#[derive(Clone)]
struct EnsembleCalcGUI {
    path: String,
    ref_name: String,
    est_name: String,
    sample_rate: u32,
    threads: usize,

    params: Parameters,

    files: Arc<Mutex<Option<TrackSetSet>>>,
    files_progress: Arc<Mutex<Option<f32>>>,
    stop: Arc<AtomicBool>,

    windows: Vec<CalcWindow>,
}

impl App for EnsembleCalcGUI {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        SidePanel::left("side_panel").max_width(420.0).default_width(420.0).show(ctx, |ui| {

            // Render progress bar or settigns
            let progress = self.files_progress.lock().unwrap().clone();
            match progress {
                Some(progress) => {
                    ui.add(ProgressBar::new(progress).animate(true).text("Loading files"));
                }
                None => self.draw_settings(ui),
            }
            
        });

        // Windows
        for window in &mut self.windows {
            window.update(ctx);
            ctx.request_repaint();
        }
    }
}

impl EnsembleCalcGUI {
    fn draw_settings(&mut self, ui: &mut Ui) {
        ui.heading("Settings");

        // Path field
        ui.label("Path to tracks");
        ui.horizontal(|ui| {
            ui.text_edit_singleline(&mut self.path);
            if ui.button("Browse").clicked() {
                if let Some(path) = FileDialog::new().pick_folder() {
                    self.path = path.to_string_lossy().to_string();
                }
            }
        });

        // Ref, est name
        ui.horizontal(|ui| {
            ui.label("Ref file name: ");
            ui.text_edit_singleline(&mut self.ref_name);
        });
        ui.horizontal(|ui| {
            ui.label("Est folder name: ");
            ui.text_edit_singleline(&mut self.est_name);
        });

        // Sample rate
        ui.horizontal(|ui| {
            ui.label("Sample Rate: ");
            let mut sr = self.sample_rate.to_string();
            if ui.text_edit_singleline(&mut sr).changed() {
                if let Ok(sr) = sr.parse() {
                    self.sample_rate = sr;
                }
            }
        });

        // Threads
        ui.add(Slider::new(&mut self.threads, 0..=16).text("Loader threads"));
        // ui.add(Slider::new(&mut self.params.tol, 1.0..=0.0).logarithmic(true).text("Solver tolerance"));
        ui.add(Slider::new(&mut self.params.max_iters, 1..=1000000000).logarithmic(true).text("Max iterations"));
        ui.add(Slider::new(&mut self.params.rng_min, -4.0..=4.0).text("Initial weights rng min value"));
        ui.add(Slider::new(&mut self.params.rng_max, -4.0..=4.0).text("Initial weights rng max value"));
        ui.add(Slider::new(&mut self.params.alpha, 0.0..=20.0).text("Alpha parameter for reflection"));
        ui.add(Slider::new(&mut self.params.gamma, 1.0..=20.0).text("Gamma for expansion"));
        ui.add(Slider::new(&mut self.params.rho, 0.0..=0.5).text("Rho for contraction"));
        ui.add(Slider::new(&mut self.params.sigma, 0.0..=1.0).text("Sigma for shrinking"));
        
        ui.add_space(32.0);
        ui.horizontal(|ui| {
            // Load the files
            if ui.button("Load files").clicked() {
                match self.load_files() {
                    Ok(_) => {},
                    Err(e) => error!("Failed loading files: {e}"),
                }
            }

            // Start
            if self.files.lock().unwrap().is_some() {
                if ui.button("Start").clicked() {
                    self.stop.store(false, Ordering::SeqCst);
                    self.windows.push(CalcWindow::new(self.params.clone(), self.windows.len(), self.files.clone(), self.stop.clone()));
                }
            }
        });

        // Show size
        if let Some(files) = self.files.lock().unwrap().as_ref() {
            ui.add_space(32.0);
            ui.label(format!("Size of tracks: {}", humansize::format_size(files.calculate_size(), DECIMAL)));
            ui.label(format!("Indicies: {:?}", files.indicies));
        }
    }

    // Load files in another thread
    fn load_files(&mut self) -> Result<(), Error> {
        let (count, rx) = crate::load_data_iter(&self.path, vec![], self.threads, &self.ref_name, &self.est_name, self.sample_rate)?;

        // Progress
        let progress = Arc::new(Mutex::new(Some(0.0)));
        self.files_progress = progress.clone();
        let output = self.files.clone();

        std::thread::spawn(move || {
            let mut out = TrackSetSet { sets: vec![], indicies: vec![] };
            for (i, item) in rx.into_iter().enumerate() {
                out.sets.push(item);
                *progress.lock().unwrap() = Some(i as f32 / count as f32);
            }
            match out.verify() {
                Ok(s) => {
                    *output.lock().unwrap() = Some(s);
                    *progress.lock().unwrap() = None;
                },
                Err(e) => error!("Failed verifying tracksets: {e}"),
            };
        });


        Ok(())
    }
}

impl Default for EnsembleCalcGUI {
    fn default() -> Self {
        Self { 
            path: Default::default(), 
            ref_name: "ref.wav".into(), 
            est_name: "est".into(),
            sample_rate: 44100,
            threads: 4,
            params: Parameters::default(),
            files: Default::default(),
            files_progress: Default::default(),
            windows: vec![],
            stop: Arc::new(AtomicBool::new(false))
        }
    }
}

#[derive(Debug, Clone)]
struct Parameters {
    pub tol: f32,
    pub max_iters: u64,
    pub rng_min: f32,
    pub rng_max: f32,
    pub extra_params: usize,

    pub alpha: f32,
    pub gamma: f32,
    pub rho: f32,
    pub sigma: f32
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            tol: 0.0005,
            max_iters: 500,
            rng_min: 0.0,
            rng_max: 1.0,
            extra_params: 1,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

#[derive(Clone)]
struct CalcWindow {
    id: usize,
    states: Vec<ObserverState>,
    state_rx: Receiver<ObserverStateWrap>,
    started: Instant,
    duration: Option<Duration>,
    indicies: Vec<String>,
    stop: Arc<AtomicBool>
}

impl CalcWindow {
    /// Create new window & thread
    pub fn new(params: Parameters, id: usize, files: Arc<Mutex<Option<TrackSetSet>>>, stop: Arc<AtomicBool>) -> CalcWindow {
        let (tx, rx) = crossbeam_channel::unbounded();
        let set = files.clone().lock().unwrap().take().unwrap();
        let indicies = set.indicies.clone();
        let stop_clone = stop.clone();

        std::thread::spawn(move || {
            // Generate mead params
            let param_count = set.indicies.len();
            let mut rng = rand::thread_rng();
            let inputs = (0..param_count + params.extra_params).into_iter().map(|_| (0..param_count).into_iter().map(|_| rng.gen_range(params.rng_min..=params.rng_max)).collect()).collect();
            let mut opt: NelderMead<Vec<f32>, f32> = NelderMead::new(inputs)
                .with_alpha(params.alpha).unwrap()
                .with_gamma(params.gamma).unwrap()
                .with_rho(params.rho).unwrap()
                .with_sigma(params.sigma).unwrap()
                .with_sd_tolerance(params.tol).unwrap();

            // Solver
            let mut problem = Problem::new(set);
            let (mut state, _) = opt.init(&mut problem, Default::default()).unwrap();
            for i in 0..params.max_iters {
                let r = opt.next_iter(&mut problem, state).unwrap();
                state = r.0;

                // Update 
                debug!("iter: {}, cost: {}, params: {:?}", state.get_iter(), state.get_cost(), state.param);
                tx.send(ObserverStateWrap::State(ObserverState { 
                    iteration: i,
                    cost: state.get_cost().to_owned().into(),
                    param: state.param.clone().unwrap()
                })).ok();

                // Stop
                if stop_clone.swap(false, Ordering::SeqCst) {
                    break;
                }
            }

            // Return files
            info!("FInished: {:?}", state);
            *files.clone().lock().unwrap() = Some(problem.take_problem().unwrap());
            tx.send(ObserverStateWrap::Finished).ok();
        });

        CalcWindow { state_rx: rx, id, states: vec![], started: Instant::now(), duration: None, indicies, stop }
    }

    pub fn update(&mut self, ctx: &Context) {
        // Check rx
        if let Ok(state) = self.state_rx.try_recv() {
            match state {
                ObserverStateWrap::Finished => self.duration = Some(self.started.elapsed()),
                ObserverStateWrap::State(state) => self.states.push(state),
            }
        }

        // Window
        Window::new(format!("Optimizer #{}", self.id)).show(ctx, |ui| {
            // Show status
            match self.duration {
                Some(duration) => ui.heading(format!("Finished! Took: {:?}, Iterations: {}", duration, self.states.len())),
                None => ui.heading(format!("Elapsed time: {:?}, Iterations: {}", self.started.elapsed(), self.states.len())),
            };

            // SDR plot
            ui.heading(format!("SDR ({:.6}):", self.states.last().map(|s| -s.cost).unwrap_or(0.0)));
            Plot::new("SDR").height(300.0).legend(Legend::default()).show(ui, |plot| {
                plot.line(Line::new(
                    self.states.iter().map(|state| {
                        [state.iteration as f64, -state.cost as f64]
                    }).collect::<PlotPoints>()
                ).name("SDR"));
            });
            
            // Params plot
            if self.states.len() > 0 {
                ui.heading("Parameters: ");
                Plot::new("Params").height(300.0).legend(Legend::default()).show(ui, |plot| {
                    for i in 0..self.states[0].param.len() {
                        plot.line(
                            Line::new(
                                self.states.iter().map(|state| {
                                    [state.iteration as f64, state.param[i] as f64]
                                }).collect::<PlotPoints>()
                            )
                            .name(self.indicies[i].clone())
                        );
                    }
                });
            }

            // Other
            ui.add_space(16.0);
            ui.label(format!("Indicies: {:?}", self.indicies));
            if self.duration.is_none() && !self.stop.load(Ordering::SeqCst) {
                if ui.button("Stop").clicked() {
                    self.stop.store(true, Ordering::SeqCst);
                }
            }

            // Save
            if self.duration.is_some() && self.states.len() > 0 {
                if ui.button("Save").clicked() {
                    let output: HashMap<String, f32> = HashMap::from_iter(self.indicies.clone().into_iter().zip(self.states.last().unwrap().param.iter().copied()));
                    let json = serde_json::to_string_pretty(&output).unwrap();
                    if let Some(path) = FileDialog::new().save_file() {
                        match std::fs::write(path, json) {
                            Ok(_) => {},
                            Err(e) => error!("Failed saving weights! {e}"),
                        }
                    }
                }
            }

        });
    }
}

#[derive(Debug, Clone)]
enum ObserverStateWrap {
    Finished,
    State(ObserverState)
}

#[derive(Debug, Clone)]
struct ObserverState {
    pub iteration: u64,
    pub cost: f32,
    pub param: Vec<f32>,
}
