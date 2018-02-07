//
//  ViewController.swift
//  Eye Tracker Test App
//
//  Created by Frederik Riedel on 2/6/18.
//  Copyright © 2018 HackMind. All rights reserved.
//

import Cocoa

class ViewController: NSViewController, NSWindowDelegate {
    
    static let numberOfCellsX = 5
    static let numberOfCellsY = 4
    
    let numberOfCells = numberOfCellsX * numberOfCellsY
    
    var currentlySelectedCell = 0
    
    var cells = Array<NSBox>()
    
    let highlightView = NSBox()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        for _ in 0...numberOfCells {
            let box = NSBox()
            box.boxType = .custom
            box.alphaValue = 1
            box.borderColor = NSColor.darkGray
            box.borderType = .lineBorder
            box.borderWidth = 2
            cells.append(box)
            self.view.addSubview(box)
        }
        
        highlightView.boxType = .custom
        highlightView.borderColor = NSColor.red
        highlightView.borderType = .lineBorder
        highlightView.borderWidth = 6
        self.view.addSubview(highlightView)
        
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) {
            self.keyDown(with: $0)
            return $0
        }
        
        layoutViews()
    }
    
    override func viewDidAppear() {
        self.view.window?.delegate = self
    }

    override var representedObject: Any? {
        didSet {
        // Update the view, if already loaded.
        }
    }
    
    func layoutViews() {
        
        let cellWidth = self.view.frame.width / CGFloat(ViewController.numberOfCellsX)
        let cellHeight = self.view.frame.height / CGFloat(ViewController.numberOfCellsY)
        
        
        
        for (index, box) in cells.enumerated() {
            let x = CGFloat(index % ViewController.numberOfCellsX) * cellWidth
            let y = CGFloat(index / ViewController.numberOfCellsX) * cellHeight
            box.frame = NSRect(x: x, y: y, width: cellWidth, height: cellHeight)
            
            if index == currentlySelectedCell {
                NSAnimationContext.runAnimationGroup({(context) -> Void in
                    context.duration = 0.1
                    highlightView.animator().frame = box.frame
                }) {
                    print("Animation done")
                }
            }
        }
    }
    
    func windowDidResize(_ notification: Notification) {
        layoutViews()
    }
    
    override func keyDown(with event: NSEvent) {
        super.keyDown(with: event)
        var newSelectedCell = currentlySelectedCell
        if event.keyCode == 123 {
            newSelectedCell-=1
        }
        if event.keyCode == 124 {
            newSelectedCell+=1
        }
        if event.keyCode == 125 {
            newSelectedCell-=ViewController.numberOfCellsX
        }
        if event.keyCode == 126 {
            newSelectedCell+=ViewController.numberOfCellsX
        }
        
        if newSelectedCell >= 0 && newSelectedCell < numberOfCells {
            currentlySelectedCell = newSelectedCell
        }
        
        layoutViews()
        
        print(event)
    }
}

